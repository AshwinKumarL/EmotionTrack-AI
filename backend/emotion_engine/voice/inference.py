import base64
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import torch
import librosa
import soundfile as sf

from backend.emotion_engine.common.base_engine import BaseEmotionEngine
from backend.emotion_engine.voice.preprocess import AudioPreprocessor
from backend.emotion_engine.voice.features import MelSpectrogramExtractor
from backend.emotion_engine.voice.model import VoiceModelConfig, EmotionCNN
from backend.emotion_engine.voice.schemas import VoiceEmotionResponse
from backend.emotion_engine.voice.train_v2 import VM2_EMOTIONS

logger = logging.getLogger(__name__)


class VoiceEmotionEngine(BaseEmotionEngine):
    """
    Inference engine for voice emotion recognition.
    Chains: raw audio -> preprocessing -> Mel spectrogram -> CNN -> label.
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: Optional[str] = None,
        preprocessor: Optional[AudioPreprocessor] = None,
        feature_extractor: Optional[MelSpectrogramExtractor] = None,
    ):
        self.model_path = model_path
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model: Optional[EmotionCNN] = None
        self.emotions = VM2_EMOTIONS
        self.preprocessor = preprocessor or AudioPreprocessor()
        self.feature_extractor = feature_extractor or MelSpectrogramExtractor()

    def load_model(self) -> None:
        if self.model_path is None or not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")

        try:
            checkpoint = torch.load(
                str(self.model_path), map_location=self.device, weights_only=False
            )
            cfg = checkpoint["model_config"]
            model_config = VoiceModelConfig(
                num_classes=cfg["num_classes"],
                input_channels=cfg["input_channels"],
                dropout_rate=cfg["dropout_rate"],
                filter_sizes=tuple(cfg["filter_sizes"]),
                kernel_sizes=tuple(cfg["kernel_sizes"]),
                pool_sizes=tuple(cfg["pool_sizes"]),
                hidden_size=cfg["hidden_size"],
            )
            self.model = EmotionCNN(model_config)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()
            self.emotions = checkpoint.get("emotions", VM2_EMOTIONS)
            logger.info(f"VM/2 model loaded from {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load voice model: {e}") from e

    def predict(self, audio_input) -> VoiceEmotionResponse:
        if self.model is None:
            self.load_model()

        waveform, sr = self._load_audio(audio_input)
        processed, _ = self.preprocessor.preprocess_audio(waveform, sr)

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, processed, self.preprocessor.target_sr)
                tmp_path = tmp.name
            mel = self.feature_extractor.extract(Path(tmp_path))
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        mel_tensor = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        mel_tensor = mel_tensor.to(self.device)

        with torch.no_grad():
            logits = self.model(mel_tensor)
            probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()

        probabilities = {}
        max_prob = -1.0
        primary = self.emotions[0]
        for idx, prob in enumerate(probs):
            label = self.emotions[idx]
            probabilities[label] = round(float(prob), 4)
            if float(prob) > max_prob:
                max_prob = float(prob)
                primary = label

        return VoiceEmotionResponse(
            primary_emotion=primary.capitalize(),
            confidence=max_prob,
            probabilities=probabilities,
            model_version="v2",
        )

    def _load_audio(self, audio_input):
        if isinstance(audio_input, (str, Path)):
            y, sr = librosa.load(str(audio_input), sr=None, mono=False)
            return y, sr
        elif isinstance(audio_input, bytes):
            audio_bytes = audio_input
        elif hasattr(audio_input, "read"):
            audio_bytes = audio_input.read()
        else:
            raise ValueError(f"Unsupported audio input type: {type(audio_input)}")

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            y, sr = librosa.load(tmp_path, sr=None, mono=False)
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)

        return y, sr

    @staticmethod
    def decode_base64_audio(audio_base64: str) -> bytes:
        return base64.b64decode(audio_base64)

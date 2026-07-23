import os
import base64
import io
import numpy as np
import pytest
import torch
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.emotion_engine.voice.model import VoiceModelConfig, EmotionCNN
from backend.emotion_engine.voice.inference import VoiceEmotionEngine
from backend.emotion_engine.voice.schemas import VoiceEmotionResponse


def _create_mock_v2_checkpoint(tmp_path, emotions=None):
    if emotions is None:
        emotions = ["happy", "sad", "angry", "fearful", "calm"]
    model_config = VoiceModelConfig(
        num_classes=5, input_channels=1, dropout_rate=0.5,
        filter_sizes=(16, 32), kernel_sizes=(3, 3), pool_sizes=(2, 2), hidden_size=64
    )
    model = EmotionCNN(model_config)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "num_classes": 5, "input_channels": 1, "dropout_rate": 0.5,
            "filter_sizes": [16, 32], "kernel_sizes": [3, 3],
            "pool_sizes": [2, 2], "hidden_size": 64
        },
        "emotions": emotions,
        "epoch": 1, "best_val_loss": 1.0, "optimizer_state_dict": {}
    }
    model_path = tmp_path / "best_voice_model_v2.pth"
    torch.save(checkpoint, str(model_path))
    return model_path


class TestVoiceEmotionEngineInit:
    def test_default_init(self):
        engine = VoiceEmotionEngine()
        assert engine.model is None
        assert engine.model_path is None
        assert engine.emotions == ["happy", "sad", "angry", "fearful", "calm"]
        assert engine.preprocessor is not None
        assert engine.feature_extractor is not None

    def test_custom_injectables(self):
        mock_preprocessor = MagicMock()
        mock_extractor = MagicMock()
        engine = VoiceEmotionEngine(
            preprocessor=mock_preprocessor,
            feature_extractor=mock_extractor
        )
        assert engine.preprocessor is mock_preprocessor
        assert engine.feature_extractor is mock_extractor


class TestLoadModel:
    def test_raises_when_path_none(self):
        engine = VoiceEmotionEngine(model_path=None)
        with pytest.raises(FileNotFoundError, match="not found"):
            engine.load_model()

    def test_raises_when_file_missing(self, tmp_path):
        engine = VoiceEmotionEngine(model_path=tmp_path / "missing.pth")
        with pytest.raises(FileNotFoundError, match="not found"):
            engine.load_model()

    def test_raises_on_corrupted_file(self, tmp_path):
        bad_path = tmp_path / "bad.pth"
        bad_path.write_text("not a checkpoint")
        engine = VoiceEmotionEngine(model_path=bad_path)
        with pytest.raises(RuntimeError, match="Failed to load"):
            engine.load_model()

    def test_raises_on_bare_state_dict(self, tmp_path):
        model_config = VoiceModelConfig(
            num_classes=5, input_channels=1, dropout_rate=0.5,
            filter_sizes=(16, 32), kernel_sizes=(3, 3), pool_sizes=(2, 2), hidden_size=64
        )
        model = EmotionCNN(model_config)
        bare_path = tmp_path / "bare.pth"
        torch.save(model.state_dict(), str(bare_path))
        engine = VoiceEmotionEngine(model_path=bare_path)
        with pytest.raises(RuntimeError):
            engine.load_model()

    def test_loads_valid_checkpoint(self, tmp_path):
        model_path = _create_mock_v2_checkpoint(tmp_path)
        engine = VoiceEmotionEngine(model_path=model_path)
        engine.load_model()
        assert engine.model is not None
        assert not engine.model.training
        assert engine.emotions == ["happy", "sad", "angry", "fearful", "calm"]


class TestLoadAudio:
    def test_load_from_bytes(self, tmp_path):
        import soundfile as sf
        audio_data = np.zeros(16000, dtype=np.float32)
        wav_path = tmp_path / "test.wav"
        sf.write(str(wav_path), audio_data, 16000)
        audio_bytes = wav_path.read_bytes()

        engine = VoiceEmotionEngine()
        waveform, sr = engine._load_audio(audio_bytes)
        assert isinstance(waveform, np.ndarray)
        assert sr > 0

    def test_load_from_path(self, tmp_path):
        import soundfile as sf
        audio_data = np.zeros(16000, dtype=np.float32)
        wav_path = tmp_path / "test.wav"
        sf.write(str(wav_path), audio_data, 16000)

        engine = VoiceEmotionEngine()
        waveform, sr = engine._load_audio(str(wav_path))
        assert isinstance(waveform, np.ndarray)

    def test_load_from_file_object(self, tmp_path):
        import soundfile as sf
        audio_data = np.zeros(16000, dtype=np.float32)
        wav_path = tmp_path / "test.wav"
        sf.write(str(wav_path), audio_data, 16000)

        engine = VoiceEmotionEngine()
        with open(wav_path, "rb") as f:
            waveform, sr = engine._load_audio(f)
        assert isinstance(waveform, np.ndarray)

    def test_unsupported_type_raises(self):
        engine = VoiceEmotionEngine()
        with pytest.raises(ValueError, match="Unsupported"):
            engine._load_audio(12345)


class TestDecodeBase64Audio:
    def test_valid_base64(self):
        original = b"hello audio"
        encoded = base64.b64encode(original).decode()
        result = VoiceEmotionEngine.decode_base64_audio(encoded)
        assert result == original

    def test_invalid_base64(self):
        with pytest.raises(Exception):
            VoiceEmotionEngine.decode_base64_audio("not-valid-base64!!!")


class TestPredict:
    def test_auto_loads_model(self, tmp_path):
        model_path = _create_mock_v2_checkpoint(tmp_path)
        engine = VoiceEmotionEngine(model_path=model_path)
        assert engine.model is None

        import soundfile as sf
        audio_data = np.zeros(48000, dtype=np.float32)
        wav_path = tmp_path / "test.wav"
        sf.write(str(wav_path), audio_data, 16000)

        result = engine.predict(str(wav_path))
        assert engine.model is not None
        assert isinstance(result, VoiceEmotionResponse)

    def test_returns_correct_response_fields(self, tmp_path):
        model_path = _create_mock_v2_checkpoint(tmp_path)
        engine = VoiceEmotionEngine(model_path=model_path)

        import soundfile as sf
        audio_data = np.sin(2 * np.pi * 440 * np.linspace(0, 3, 48000)).astype(np.float32)
        wav_path = tmp_path / "tone.wav"
        sf.write(str(wav_path), audio_data, 16000)

        result = engine.predict(str(wav_path))
        assert isinstance(result.primary_emotion, str)
        assert isinstance(result.confidence, float)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.probabilities, dict)
        assert result.model_version == "v2"

    def test_probabilities_sum_to_one(self, tmp_path):
        model_path = _create_mock_v2_checkpoint(tmp_path)
        engine = VoiceEmotionEngine(model_path=model_path)

        import soundfile as sf
        audio_data = np.random.randn(48000).astype(np.float32)
        wav_path = tmp_path / "noise.wav"
        sf.write(str(wav_path), audio_data, 16000)

        result = engine.predict(str(wav_path))
        total = sum(result.probabilities.values())
        assert total == pytest.approx(1.0, abs=0.01)

    def test_temp_files_cleaned_up(self, tmp_path):
        model_path = _create_mock_v2_checkpoint(tmp_path)
        engine = VoiceEmotionEngine(model_path=model_path)

        import soundfile as sf
        audio_data = np.zeros(48000, dtype=np.float32)
        wav_path = tmp_path / "test.wav"
        sf.write(str(wav_path), audio_data, 16000)

        temp_dir = Path(os.environ.get("TEMP", os.environ.get("TMP", "/tmp")))
        before = set(temp_dir.glob("*.wav"))
        engine.predict(str(wav_path))
        after = set(temp_dir.glob("*.wav"))
        new_files = after - before
        assert len(new_files) == 0

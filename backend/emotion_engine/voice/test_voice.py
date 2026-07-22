import argparse
import sys
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F
import librosa

# Setup path manipulation so the script can be run directly from any directory
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.emotion_engine.voice.loader import VoiceEngineError
from backend.emotion_engine.voice.dataset import LabelEncoder
from backend.emotion_engine.voice.model import VoiceModelConfig, EmotionCNN
from backend.emotion_engine.voice.preprocess import AudioPreprocessor, PeakNormalizer
from backend.emotion_engine.voice.features import MelSpectrogramExtractor

# Base target directory for manual testing assets
TEST_DIR = Path("tests/test_voice")
TEST_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# Audio Recording Module
# =====================================================================

def record_audio(duration: float = 3.0, sample_rate: int = 16000) -> Path:
    """
    Records audio directly from the user's microphone using sounddevice
    and saves it to tests/test_voice/temp_recording.wav.
    
    Args:
        duration (float): Recording length in seconds.
        sample_rate (int): Sampling frequency in Hz.
        
    Returns:
        Path: Path to the saved temporary recording.
    """
    try:
        import sounddevice as sd
    except ImportError:
        print("Error: The 'sounddevice' package is required for microphone recording.", file=sys.stderr)
        print("Please ensure you installed dependencies with: pip install -r requirements.txt", file=sys.stderr)
        sys.exit(1)
        
    print(f"\n[Microphone Mode] Recording will start in 1 second. Prepare to speak...")
    time.sleep(1)
    
    print(f"=== RECORDING STARTED ({duration} seconds) ===")
    try:
        # Record 1D mono audio float32 array
        recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
        sd.wait()  # Block execution until finished
    except Exception as e:
        print(f"Error accessing microphone: {e}", file=sys.stderr)
        sys.exit(1)
    print("=== RECORDING FINISHED ===")
    
    save_path = TEST_DIR / "temp_recording.wav"
    try:
        # Squeeze channel dim so that shape is 1D (samples,)
        audio_data = np.squeeze(recording)
        sf.write(str(save_path), audio_data, sample_rate)
        return save_path
    except Exception as e:
        print(f"Error saving temporary microphone audio: {e}", file=sys.stderr)
        sys.exit(1)


# =====================================================================
# Model Weight Loader
# =====================================================================

def load_model(model_path: Path, num_classes: int, device: torch.device) -> EmotionCNN:
    """
    Recreates the CNN architecture and loads the saved model state dict weights.
    Used for VM/1 checkpoints (bare state_dict format).
    
    Args:
        model_path (Path): Path to the saved weights file (.pth).
        num_classes (int): Number of target emotion categories.
        device (torch.device): Device to run the model on.
        
    Returns:
        EmotionCNN: The instantiated and loaded model in eval mode.
    """
    if not model_path.exists():
        print(f"Error: Trained model checkpoint not found at: '{model_path}'", file=sys.stderr)
        print("Please train the model first by running: python backend/emotion_engine/voice/train.py --train", file=sys.stderr)
        sys.exit(1)
        
    try:
        model_config = VoiceModelConfig(
            num_classes=num_classes,
            input_channels=1,
            dropout_rate=0.5,
            filter_sizes=(32, 64, 128),
            kernel_sizes=(3, 3, 3),
            pool_sizes=(2, 2, 2),
            hidden_size=256
        )
        model = EmotionCNN(model_config)
        
        state_dict = torch.load(str(model_path), map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading trained model weights: {e}", file=sys.stderr)
        sys.exit(1)


def load_model_v2(model_path: Path, device: torch.device) -> EmotionCNN:
    """
    Loads the VM/2 model from a metadata-rich checkpoint.
    
    VM/2 checkpoints contain model_config, emotions, epoch, and optimizer
    state alongside model weights, unlike VM/1's bare state_dict format.
    
    Args:
        model_path (Path): Path to best_voice_model_v2.pth.
        device (torch.device): Target device.
        
    Returns:
        EmotionCNN: Loaded model in eval mode.
    """
    if not model_path.exists():
        print(f"Error: VM/2 checkpoint not found at: '{model_path}'", file=sys.stderr)
        print("Please train VM/2 first by running: python backend/emotion_engine/voice/train_v2.py --train", file=sys.stderr)
        sys.exit(1)
        
    try:
        checkpoint = torch.load(str(model_path), map_location=device)
        cfg = checkpoint["model_config"]
        model_config = VoiceModelConfig(
            num_classes=cfg["num_classes"],
            input_channels=cfg["input_channels"],
            dropout_rate=cfg["dropout_rate"],
            filter_sizes=tuple(cfg["filter_sizes"]),
            kernel_sizes=tuple(cfg["kernel_sizes"]),
            pool_sizes=tuple(cfg["pool_sizes"]),
            hidden_size=cfg["hidden_size"]
        )
        model = EmotionCNN(model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading VM/2 model weights: {e}", file=sys.stderr)
        sys.exit(1)


# =====================================================================
# Prediction Loop
# =====================================================================

def predict_emotion(
    audio_path: Path,
    model: EmotionCNN,
    label_encoder: LabelEncoder,
    device: torch.device
) -> None:
    """
    Loads, preprocesses, extracts features, runs inference, and prints predictions.
    
    Args:
        audio_path (Path): File path to the target audio.
        model (EmotionCNN): The loaded model.
        label_encoder (LabelEncoder): The label encoder.
        device (torch.device): Device to run inference on.
    """
    if not audio_path.exists():
        print(f"Error: Target audio file not found: {audio_path}", file=sys.stderr)
        sys.exit(1)
        
    try:
        # 1. Load raw audio
        # Using librosa.load with sr=None preserves native sampling rate
        y, sr = librosa.load(str(audio_path), sr=None)
    except Exception as e:
        print(f"Error loading audio file '{audio_path}': {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Preprocess waveform (Mono conversion, resampling, silence trim, peak normalization, pad/trim to 3.0s)
    try:
        preprocessor = AudioPreprocessor(
            target_sr=16000,
            target_duration=3.0,
            normalizer=PeakNormalizer(target_peak=1.0),
            silence_trimming_enabled=True,
            silence_threshold_db=30.0,
            mono_conversion_enabled=True
        )
        processed_y, metrics = preprocessor.preprocess_audio(y, sr)
    except Exception as e:
        print(f"Error preprocessing audio: {e}", file=sys.stderr)
        sys.exit(1)

    # Save preprocessed audio to temporary file
    preprocessed_path = TEST_DIR / "preprocessed.wav"
    try:
        sf.write(str(preprocessed_path), processed_y, 16000)
    except Exception as e:
        print(f"Error saving preprocessed audio: {e}", file=sys.stderr)
        sys.exit(1)

    # 3. Extract Features (Mel Spectrogram)
    try:
        extractor = MelSpectrogramExtractor(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            n_mels=128
        )
        features = extractor.extract(preprocessed_path)
    except Exception as e:
        print(f"Error extracting features: {e}", file=sys.stderr)
        sys.exit(1)

    # 4. Run Inference using CNN
    try:
        # Convert numpy features array of shape (128, 94) to Torch Tensor (1, 1, 128, 94)
        feature_tensor = torch.from_numpy(features).float().unsqueeze(0).unsqueeze(0)
        feature_tensor = feature_tensor.to(device)
        
        with torch.no_grad():
            logits = model(feature_tensor)
            # Apply softmax to calculate probability scores
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    except Exception as e:
        print(f"Error during model inference: {e}", file=sys.stderr)
        sys.exit(1)

    # 5. Format and Print Outputs
    predicted_idx = int(np.argmax(probs))
    predicted_emotion = label_encoder.decode(predicted_idx)
    confidence = probs[predicted_idx]

    # Map all class probabilities
    emotions = label_encoder.emotions
    emotion_probs = []
    for idx, prob in enumerate(probs):
        emotion_probs.append((emotions[idx].capitalize(), prob))
        
    # Sort in descending order of probability
    emotion_probs.sort(key=lambda x: x[1], reverse=True)

    print("\n================================")
    print("Predicted Emotion")
    print(predicted_emotion.capitalize())
    print("\nConfidence")
    print(f"{confidence * 100:.2f}%")
    print("\nClass Probabilities")
    for emo, p in emotion_probs:
        print(f"{emo:<11} {p * 100:.2f}%")
    print("================================\n")


# =====================================================================
# CLI Entry Point
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Voice Emotion Recognition Inference Tester")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--mic",
        action="store_true",
        help="Record 3 seconds of audio from the microphone for emotion prediction."
    )
    group.add_argument(
        "--file",
        type=str,
        help="Path to an existing WAV audio file for emotion prediction."
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["VM1", "VM2"],
        default="VM1",
        help="Select which voice model to use. VM1 = 8-class (default), VM2 = 5-class."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Configure path defaults
    root_dir = Path(__file__).resolve().parent.parent.parent.parent
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Select model version
    if args.model == "VM2":
        # VM/2: 5-class conversational emotion model
        vm2_emotions = ["happy", "sad", "angry", "fearful", "calm"]
        model_save = root_dir / "models" / "best_voice_model_v2.pth"
        label_encoder = LabelEncoder(emotions=vm2_emotions)
        model = load_model_v2(model_save, device)
        print(f"[Using Voice Model V2 — 5-class]")
    else:
        # VM/1: 8-class default (original behavior)
        model_save = root_dir / "models" / "best_voice_model.pth"
        label_encoder = LabelEncoder()
        num_classes = label_encoder.num_classes()
        model = load_model(model_save, num_classes, device)
        print(f"[Using Voice Model V1 — 8-class]")

    if args.mic:
        audio_path = record_audio(duration=3.0, sample_rate=16000)
    else:
        audio_path = Path(args.file)

    predict_emotion(audio_path, model, label_encoder, device)


if __name__ == "__main__":
    main()

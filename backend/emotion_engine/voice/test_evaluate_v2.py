import csv
import json
import numpy as np
import pytest
import torch
from pathlib import Path

from backend.emotion_engine.voice.model import VoiceModelConfig, EmotionCNN
from backend.emotion_engine.voice.evaluate import ModelLoadError, DatasetLoadError
from backend.emotion_engine.voice.evaluate_v2 import load_v2_model, run_v2_evaluation


def _create_mock_v2_dataset(tmp_path):
    """Creates a minimal 5-class dataset with synthetic feature files."""
    features_dir = tmp_path / "features"
    features_dir.mkdir()

    emotions = ["happy", "sad", "angry", "fearful", "calm", "happy", "sad"]
    npy_files = []
    for idx, emotion in enumerate(emotions):
        data = np.zeros((128, 94), dtype=np.float32)
        npy_path = features_dir / f"sample_{idx}.npy"
        np.save(str(npy_path), data)
        npy_files.append(npy_path)

    index_path = tmp_path / "feature_index_v2.csv"
    with open(index_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_path", "original_audio_path", "processed_audio_path", "dataset", "speaker_id", "emotion", "feature_type", "feature_shape"])
        for idx, emotion in enumerate(emotions):
            writer.writerow([
                str(npy_files[idx]),
                "raw.wav",
                "proc.wav",
                "RAVDESS",
                f"actor_{idx}",
                emotion,
                "mel",
                "(128, 94)"
            ])

    return index_path, features_dir


def _create_mock_v2_checkpoint(tmp_path, emotions=None):
    """Creates a metadata-rich VM/2 checkpoint for testing."""
    if emotions is None:
        emotions = ["happy", "sad", "angry", "fearful", "calm"]

    model_config = VoiceModelConfig(
        num_classes=5,
        input_channels=1,
        dropout_rate=0.5,
        filter_sizes=(16, 32),
        kernel_sizes=(3, 3),
        pool_sizes=(2, 2),
        hidden_size=64
    )
    model = EmotionCNN(model_config)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "num_classes": model.config.num_classes,
            "input_channels": model.config.input_channels,
            "dropout_rate": model.config.dropout_rate,
            "filter_sizes": list(model.config.filter_sizes),
            "kernel_sizes": list(model.config.kernel_sizes),
            "pool_sizes": list(model.config.pool_sizes),
            "hidden_size": model.config.hidden_size
        },
        "emotions": emotions,
        "epoch": 1,
        "best_val_loss": 1.234,
        "optimizer_state_dict": {}
    }
    model_path = tmp_path / "best_voice_model_v2.pth"
    torch.save(checkpoint, str(model_path))
    return model_path


class TestLoadV2Model:
    def test_loads_metadata_rich_checkpoint(self, tmp_path):
        """Successfully loads a valid VM/2 checkpoint and returns eval-mode model."""
        model_path = _create_mock_v2_checkpoint(tmp_path)
        model = load_v2_model(model_path, torch.device("cpu"))

        assert isinstance(model, EmotionCNN)
        assert model.config.num_classes == 5
        assert not model.training

    def test_raises_on_missing_file(self, tmp_path):
        """Raises ModelLoadError when checkpoint file does not exist."""
        with pytest.raises(ModelLoadError, match="not found"):
            load_v2_model(tmp_path / "nonexistent.pth", torch.device("cpu"))

    def test_raises_on_bare_state_dict(self, tmp_path):
        """Raises ModelLoadError when given a VM/1 bare state_dict format."""
        model_config = VoiceModelConfig(
            num_classes=5, input_channels=1, dropout_rate=0.5,
            filter_sizes=(16, 32), kernel_sizes=(3, 3), pool_sizes=(2, 2), hidden_size=64
        )
        model = EmotionCNN(model_config)
        bare_path = tmp_path / "bare_state_dict.pth"
        torch.save(model.state_dict(), str(bare_path))

        with pytest.raises(ModelLoadError, match="missing expected key"):
            load_v2_model(bare_path, torch.device("cpu"))


class TestRunV2Evaluation:
    def test_evaluation_completes_successfully(self, tmp_path):
        """Full evaluation pipeline completes and returns metrics dict."""
        index_path, features_dir = _create_mock_v2_dataset(tmp_path)
        model_path = _create_mock_v2_checkpoint(tmp_path)
        output_dir = tmp_path / "reports"

        metrics = run_v2_evaluation(
            model_path=model_path,
            index_path=index_path,
            features_dir=features_dir,
            output_dir=output_dir,
            validation_split=0.4,
            random_seed=42,
            device_str="cpu"
        )

        assert "accuracy" in metrics
        assert "precision_macro" in metrics
        assert "recall_macro" in metrics
        assert "f1_macro" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_evaluation_writes_all_reports(self, tmp_path):
        """Evaluation saves JSON, per-class JSON, TXT, and PNG reports."""
        index_path, features_dir = _create_mock_v2_dataset(tmp_path)
        model_path = _create_mock_v2_checkpoint(tmp_path)
        output_dir = tmp_path / "reports"

        run_v2_evaluation(
            model_path=model_path,
            index_path=index_path,
            features_dir=features_dir,
            output_dir=output_dir,
            validation_split=0.4,
            random_seed=42,
            device_str="cpu"
        )

        assert (output_dir / "evaluation_metrics.json").exists()
        assert (output_dir / "per_class_metrics.json").exists()
        assert (output_dir / "evaluation_report.txt").exists()
        assert (output_dir / "confusion_matrix.png").exists()
        assert (output_dir / "confusion_matrix.png").stat().st_size > 0

    def test_per_class_metrics_structure(self, tmp_path):
        """Per-class metrics JSON contains all 5 VM/2 emotions with correct fields."""
        index_path, features_dir = _create_mock_v2_dataset(tmp_path)
        model_path = _create_mock_v2_checkpoint(tmp_path)
        output_dir = tmp_path / "reports"

        run_v2_evaluation(
            model_path=model_path,
            index_path=index_path,
            features_dir=features_dir,
            output_dir=output_dir,
            validation_split=0.4,
            random_seed=42,
            device_str="cpu"
        )

        per_class_path = output_dir / "per_class_metrics.json"
        with open(per_class_path, "r", encoding="utf-8") as f:
            per_class = json.load(f)

        expected_emotions = {"happy", "sad", "angry", "fearful", "calm"}
        assert set(per_class.keys()) == expected_emotions
        for emotion in expected_emotions:
            assert "precision" in per_class[emotion]
            assert "recall" in per_class[emotion]
            assert "f1" in per_class[emotion]
            assert "support" in per_class[emotion]

    def test_raises_on_missing_index(self, tmp_path):
        """Raises DatasetLoadError when index file is missing."""
        model_path = _create_mock_v2_checkpoint(tmp_path)
        with pytest.raises(DatasetLoadError):
            run_v2_evaluation(
                model_path=model_path,
                index_path=tmp_path / "missing.csv",
                features_dir=tmp_path,
                output_dir=tmp_path / "reports",
                device_str="cpu"
            )

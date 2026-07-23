import csv
import json
import numpy as np
import pytest
import torch
from pathlib import Path

from backend.emotion_engine.voice.dataset import VoiceDatasetConfig, EmotionDataset
from backend.emotion_engine.voice.model import VoiceModelConfig, EmotionCNN
from backend.emotion_engine.voice.train import VoiceTrainingConfig, TrainingError
from backend.emotion_engine.voice.train_v2 import (
    VM2_EMOTIONS,
    generate_v2_index_if_needed,
    compute_class_weights,
    run_training_v2,
)


@pytest.fixture
def mock_v2_dataset_and_paths(tmp_path):
    """Creates a temporary 5-class dataset with synthetic feature files."""
    features_dir = tmp_path / "features"
    features_dir.mkdir()

    emotions = ["happy", "sad", "angry", "fearful", "calm", "happy", "sad", "angry", "fearful", "calm"]
    npy_files = []
    for idx, emotion in enumerate(emotions):
        data = np.random.randn(128, 94).astype(np.float32)
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

    model_save_path = tmp_path / "best_model_v2.pth"
    log_dir = tmp_path / "logs"
    return index_path, features_dir, model_save_path, log_dir


@pytest.fixture
def v2_train_config(mock_v2_dataset_and_paths):
    _, _, model_save_path, _ = mock_v2_dataset_and_paths
    return VoiceTrainingConfig(
        batch_size=4,
        epochs=2,
        learning_rate=0.001,
        weight_decay=0.0001,
        validation_split=0.2,
        random_seed=42,
        device="cpu",
        num_workers=0,
        model_save_path=model_save_path,
        patience=5
    )


def _load_dataset(index_path, features_dir, emotions=None):
    config = VoiceDatasetConfig(index_path=index_path, features_dir=features_dir, emotions=emotions)
    return EmotionDataset(config=config)


def _build_model(num_classes=5):
    return EmotionCNN(VoiceModelConfig(
        num_classes=num_classes,
        input_channels=1,
        dropout_rate=0.5,
        filter_sizes=(16, 32),
        kernel_sizes=(3, 3),
        pool_sizes=(2, 2),
        hidden_size=64
    ))


class TestGenerateV2Index:
    def test_generates_filtered_index(self, tmp_path):
        """Creates a filtered index with only allowed emotions."""
        source = tmp_path / "feature_index.csv"
        target = tmp_path / "feature_index_v2.csv"

        rows = [
            ["happy", "feature1.npy"],
            ["sad", "feature2.npy"],
            ["neutral", "feature3.npy"],
            ["angry", "feature4.npy"],
            ["disgust", "feature5.npy"],
        ]
        with open(source, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["emotion", "feature_path"])
            writer.writerows(rows)

        count = generate_v2_index_if_needed(source, target, VM2_EMOTIONS)
        assert count == 3  # happy, sad, angry are in VM2_EMOTIONS; neutral and disgust are not

        with open(target, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            saved_emotions = [row["emotion"] for row in reader]
        assert saved_emotions == ["happy", "sad", "angry"]
        assert len(saved_emotions) == 3

    def test_skips_if_already_exists(self, tmp_path):
        """Returns -1 if target index already exists."""
        source = tmp_path / "source.csv"
        target = tmp_path / "target.csv"
        source.write_text("emotion\nhappy\n")
        target.write_text("emotion\nhappy\n")

        result = generate_v2_index_if_needed(source, target, VM2_EMOTIONS)
        assert result == -1

    def test_raises_on_missing_source(self, tmp_path):
        """Raises TrainingError if source index is missing."""
        with pytest.raises(TrainingError, match="not found"):
            generate_v2_index_if_needed(
                tmp_path / "missing.csv",
                tmp_path / "target.csv",
                VM2_EMOTIONS
            )


class TestComputeClassWeights:
    def test_weights_shape_and_values(self, mock_v2_dataset_and_paths):
        """Weights tensor has correct shape and inverse-frequency properties."""
        index_path, features_dir, _, _ = mock_v2_dataset_and_paths
        dataset = _load_dataset(index_path, features_dir, VM2_EMOTIONS)

        from torch.utils.data import random_split
        generator = torch.Generator().manual_seed(42)
        total = len(dataset)
        val_len = int(total * 0.2)
        train_len = total - val_len
        train_subset, _ = random_split(dataset, [train_len, val_len], generator=generator)

        num_classes = 5
        weights = compute_class_weights(dataset, train_subset, num_classes, torch.device("cpu"))

        assert weights.shape == (num_classes,)
        assert all(w > 0 for w in weights)
        # Rarer classes should have higher weights
        min_weight = weights.min()
        max_weight = weights.max()
        assert max_weight >= min_weight


class TestRunTrainingV2:
    def test_smoke_test_completes(self, mock_v2_dataset_and_paths, v2_train_config):
        """Smoke test (1 epoch, 1 batch) completes without errors."""
        index_path, features_dir, _, log_dir = mock_v2_dataset_and_paths
        dataset = _load_dataset(index_path, features_dir, VM2_EMOTIONS)
        model = _build_model(num_classes=5)

        history = run_training_v2(model, dataset, v2_train_config, log_dir, smoke_test=True)

        assert len(history["train_loss"]) == 1
        assert len(history["val_loss"]) == 1
        assert history["best_epoch"] == 1

    def test_weight_updates_after_step(self, mock_v2_dataset_and_paths, v2_train_config):
        """Model weights update after a training step."""
        index_path, features_dir, _, log_dir = mock_v2_dataset_and_paths
        dataset = _load_dataset(index_path, features_dir, VM2_EMOTIONS)
        model = _build_model(num_classes=5)

        initial_weights = model.classifier[1].weight.clone().detach()
        run_training_v2(model, dataset, v2_train_config, log_dir, smoke_test=True)
        updated_weights = model.classifier[1].weight.clone().detach()

        assert not torch.equal(initial_weights, updated_weights)

    def test_metadata_rich_checkpoint_saved(self, mock_v2_dataset_and_paths, v2_train_config):
        """Full training saves a metadata-rich checkpoint."""
        index_path, features_dir, model_save_path, log_dir = mock_v2_dataset_and_paths
        dataset = _load_dataset(index_path, features_dir, VM2_EMOTIONS)
        model = _build_model(num_classes=5)

        run_training_v2(model, dataset, v2_train_config, log_dir, smoke_test=False)

        assert model_save_path.exists()
        checkpoint = torch.load(str(model_save_path), map_location="cpu")
        assert "model_state_dict" in checkpoint
        assert "model_config" in checkpoint
        assert "emotions" in checkpoint
        assert checkpoint["emotions"] == VM2_EMOTIONS

    def test_training_log_json_saved(self, mock_v2_dataset_and_paths, v2_train_config):
        """Training log JSON is written to the log directory."""
        index_path, features_dir, model_save_path, log_dir = mock_v2_dataset_and_paths
        dataset = _load_dataset(index_path, features_dir, VM2_EMOTIONS)
        model = _build_model(num_classes=5)

        run_training_v2(model, dataset, v2_train_config, log_dir, smoke_test=False)

        log_path = log_dir / "training_log.json"
        assert log_path.exists()
        with open(log_path, "r", encoding="utf-8") as f:
            log = json.load(f)
        assert "best_epoch" in log
        assert "total_epochs_run" in log
        assert len(log["train_loss"]) > 0

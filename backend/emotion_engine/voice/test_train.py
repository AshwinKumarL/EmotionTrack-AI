import csv
import numpy as np
import pytest
import torch
from pathlib import Path

from backend.emotion_engine.voice.dataset import VoiceDatasetConfig, EmotionDataset
from backend.emotion_engine.voice.model import VoiceModelConfig, EmotionCNN
from backend.emotion_engine.voice.train import (
    VoiceTrainingConfig,
    setup_data_loaders,
    run_training,
    TrainingConfigurationError,
    TrainingError
)


@pytest.fixture
def mock_dataset_and_paths(tmp_path):
    """
    Creates a temporary dataset index and synthetic feature files for testing.
    """
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    
    # Create 10 synthetic feature files representing standard 8 emotions plus padding
    npy_files = []
    emotions = ["happy", "sad", "neutral", "calm", "angry", "fearful", "disgust", "surprised", "happy", "sad"]
    
    for idx, emotion in enumerate(emotions):
        data = np.random.randn(128, 94).astype(np.float32)
        npy_path = features_dir / f"sample_{idx}.npy"
        np.save(str(npy_path), data)
        npy_files.append(npy_path)
        
    index_path = tmp_path / "feature_index.csv"
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
            
    model_save_path = tmp_path / "best_model.pth"
    return index_path, features_dir, model_save_path


def test_training_config_validation(tmp_path):
    """Verify that VoiceTrainingConfig handles bounds checking correctly."""
    valid_path = tmp_path / "best_model.pth"
    
    # 1. Invalid batch size
    with pytest.raises(TrainingConfigurationError):
        VoiceTrainingConfig(
            batch_size=-10,
            epochs=10,
            learning_rate=0.001,
            weight_decay=0.0,
            validation_split=0.2,
            random_seed=42,
            device="cpu",
            num_workers=0,
            model_save_path=valid_path,
            patience=5
        )
        
    # 2. Invalid validation split bounds
    with pytest.raises(TrainingConfigurationError):
        VoiceTrainingConfig(
            batch_size=16,
            epochs=10,
            learning_rate=0.001,
            weight_decay=0.0,
            validation_split=1.5,  # must be between 0 and 1
            random_seed=42,
            device="cpu",
            num_workers=0,
            model_save_path=valid_path,
            patience=5
        )
        
    # 3. Invalid model save path type
    with pytest.raises(TrainingConfigurationError):
        VoiceTrainingConfig(
            batch_size=16,
            epochs=10,
            learning_rate=0.001,
            weight_decay=0.0,
            validation_split=0.2,
            random_seed=42,
            device="cpu",
            num_workers=0,
            model_save_path="string_instead_of_path_object.pth",
            patience=5
        )

    # 4. Invalid patience parameter
    with pytest.raises(TrainingConfigurationError):
        VoiceTrainingConfig(
            batch_size=16,
            epochs=10,
            learning_rate=0.001,
            weight_decay=0.0,
            validation_split=0.2,
            random_seed=42,
            device="cpu",
            num_workers=0,
            model_save_path=valid_path,
            patience=-1  # must be positive
        )


def test_dataset_splitting_and_loaders(mock_dataset_and_paths):
    """Verify that dataset splits are consistent and Loader counts map to configuration."""
    index_path, features_dir, model_save_path = mock_dataset_and_paths
    
    dataset_config = VoiceDatasetConfig(index_path=index_path, features_dir=features_dir)
    dataset = EmotionDataset(dataset_config)
    
    train_config = VoiceTrainingConfig(
        batch_size=2,
        epochs=2,
        learning_rate=0.001,
        weight_decay=0.0001,
        validation_split=0.3,  # 3 out of 10 samples for validation, 7 for train
        random_seed=123,
        device="cpu",
        num_workers=0,
        model_save_path=model_save_path,
        patience=5
    )
    
    train_loader, val_loader = setup_data_loaders(dataset, train_config)
    
    assert len(train_loader.dataset) == 7
    assert len(val_loader.dataset) == 3
    
    # Assert reproducible seed behavior
    train_indices_1 = [idx for idx in train_loader.dataset.indices]
    
    # Setup second time with same config
    train_loader_2, _ = setup_data_loaders(dataset, train_config)
    train_indices_2 = [idx for idx in train_loader_2.dataset.indices]
    
    assert train_indices_1 == train_indices_2


def test_train_loop_step(mock_dataset_and_paths):
    """Verify that running a single batch updates model weights."""
    index_path, features_dir, model_save_path = mock_dataset_and_paths
    
    dataset_config = VoiceDatasetConfig(index_path=index_path, features_dir=features_dir)
    dataset = EmotionDataset(dataset_config)
    
    model_config = VoiceModelConfig(
        num_classes=8,
        input_channels=1,
        dropout_rate=0.5,
        filter_sizes=(16, 32),
        kernel_sizes=(3, 3),
        pool_sizes=(2, 2),
        hidden_size=64
    )
    model = EmotionCNN(model_config)
    
    train_config = VoiceTrainingConfig(
        batch_size=2,
        epochs=1,
        learning_rate=0.01,
        weight_decay=0.0,
        validation_split=0.2,
        random_seed=42,
        device="cpu",
        num_workers=0,
        model_save_path=model_save_path,
        patience=5
    )
    
    # Capture initial weights of a model parameter to check update
    initial_weights = model.classifier[1].weight.clone().detach()
    
    # Run a smoke test (1 epoch, 1 batch)
    history = run_training(model, dataset, train_config, smoke_test=True)
    
    # Capture weights after training step
    updated_weights = model.classifier[1].weight.clone().detach()
    
    # Weights must differ after optimizer step
    assert not torch.equal(initial_weights, updated_weights), "Weights did not update after forward/backward step."
    assert len(history["train_loss"]) == 1
    assert len(history["val_loss"]) == 1


def test_checkpoint_saving(mock_dataset_and_paths):
    """Verify that model checkpointing writes state dict correctly on best loss."""
    index_path, features_dir, model_save_path = mock_dataset_and_paths
    
    dataset_config = VoiceDatasetConfig(index_path=index_path, features_dir=features_dir)
    dataset = EmotionDataset(dataset_config)
    
    model_config = VoiceModelConfig(
        num_classes=8,
        input_channels=1,
        dropout_rate=0.5,
        filter_sizes=(16,),
        kernel_sizes=(3,),
        pool_sizes=(2,),
        hidden_size=32
    )
    model = EmotionCNN(model_config)
    
    # In training runs (not smoke test), it writes checkpoints to model_save_path
    train_config = VoiceTrainingConfig(
        batch_size=4,
        epochs=1,
        learning_rate=0.001,
        weight_decay=0.0,
        validation_split=0.2,
        random_seed=42,
        device="cpu",
        num_workers=0,
        model_save_path=model_save_path,
        patience=5
    )
    
    # Execute normal training (epochs=1)
    run_training(model, dataset, train_config, smoke_test=False)
    
    # Confirm file exists and is populated
    assert model_save_path.exists(), "Model checkpoint was not written to disk."
    assert model_save_path.stat().st_size > 0
    
    # Verify we can load the state dict back
    state_dict = torch.load(str(model_save_path))
    assert "classifier.1.weight" in state_dict or "classifier.4.weight" in state_dict


def test_early_stopping_triggered(mock_dataset_and_paths):
    """Verify that training loop terminates early when validation loss fails to improve."""
    index_path, features_dir, model_save_path = mock_dataset_and_paths
    
    dataset_config = VoiceDatasetConfig(index_path=index_path, features_dir=features_dir)
    dataset = EmotionDataset(dataset_config)
    
    model_config = VoiceModelConfig(
        num_classes=8,
        input_channels=1,
        dropout_rate=0.5,
        filter_sizes=(8,),
        kernel_sizes=(3,),
        pool_sizes=(2,),
        hidden_size=16
    )
    model = EmotionCNN(model_config)
    
    # Create configuration with low patience
    train_config = VoiceTrainingConfig(
        batch_size=4,
        epochs=10,
        learning_rate=0.001,
        weight_decay=0.0,
        validation_split=0.2,
        random_seed=42,
        device="cpu",
        num_workers=0,
        model_save_path=model_save_path,
        patience=2  # stop after 2 epochs without improvement
    )

    # We run training. Because the dataset has dummy identical samples (all zeros),
    # validation loss will plateau quickly, triggering early stopping.
    history = run_training(model, dataset, train_config, smoke_test=False)
    
    # The history epochs should be less than configured 10
    assert len(history["train_loss"]) < 10, f"Early stopping did not trigger. Epochs executed: {len(history['train_loss'])}"

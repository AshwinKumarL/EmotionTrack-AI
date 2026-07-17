import csv
import numpy as np
import pytest
import torch
from pathlib import Path

from backend.emotion_engine.voice.dataset import (
    VoiceDatasetConfig,
    LabelEncoder,
    EmotionDataset,
    DatasetConfigError,
    InvalidLabelError,
    DatasetIOError,
)


def test_label_encoder_default():
    """Tests the default behavior of the LabelEncoder."""
    encoder = LabelEncoder()
    assert encoder.num_classes() == 8
    assert "neutral" in encoder.emotions
    
    # Test encoding
    assert encoder.encode("neutral") == 0
    assert encoder.encode("Calm") == 1  # test case-insensitivity
    assert encoder.encode("  HAPPY  ") == 2  # test whitespace stripping
    
    # Test decoding
    assert encoder.decode(0) == "neutral"
    assert encoder.decode(1) == "calm"
    assert encoder.decode(7) == "surprised"


def test_label_encoder_custom():
    """Tests custom emotion list configuration and de-duplication."""
    encoder = LabelEncoder(emotions=["happy", "sad", "Happy", "angry"])
    assert encoder.num_classes() == 3
    assert encoder.emotions == ["happy", "sad", "angry"]
    
    assert encoder.encode("happy") == 0
    assert encoder.encode("sad") == 1
    assert encoder.encode("angry") == 2
    
    assert encoder.decode(0) == "happy"


def test_label_encoder_validation():
    """Tests that LabelEncoder correctly raises exceptions for invalid parameters/calls."""
    encoder = LabelEncoder()
    
    # Test invalid string encoding
    with pytest.raises(InvalidLabelError):
        encoder.encode("excited")  # not in the standard set
        
    with pytest.raises(InvalidLabelError):
        encoder.encode(123)  # type mismatch
        
    # Test invalid index decoding
    with pytest.raises(InvalidLabelError):
        encoder.decode(-1)
        
    with pytest.raises(InvalidLabelError):
        encoder.decode(10)  # out of range
        
    with pytest.raises(InvalidLabelError):
        encoder.decode("0")  # type mismatch

    # Test invalid initialization
    with pytest.raises(DatasetConfigError):
        LabelEncoder(emotions=[])  # empty list
        
    with pytest.raises(DatasetConfigError):
        LabelEncoder(emotions=["happy", 123])  # type mismatch in list


def test_dataset_config_validation():
    """Tests validation constraints in VoiceDatasetConfig."""
    # Invalid index_path type
    with pytest.raises(DatasetConfigError):
        VoiceDatasetConfig(index_path="not_a_path_object.csv", features_dir=Path("."))
        
    # Invalid features_dir type
    with pytest.raises(DatasetConfigError):
        VoiceDatasetConfig(index_path=Path("."), features_dir="not_a_path_object")


@pytest.fixture
def mock_dataset_environment(tmp_path):
    """
    Creates a temporary file structure containing:
    - 3 synthetic .npy feature files
    - A mock feature_index.csv file mapping to these files.
    """
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    
    # Create 3 synthetic Mel Spectrogram arrays of shape (128, 94)
    # Using different values to verify correct file loading
    npy_files = []
    emotions = ["happy", "sad", "neutral"]
    
    for idx, emotion in enumerate(emotions):
        # Shape: (128, 94)
        data = np.full((128, 94), fill_value=float(idx), dtype=np.float32)
        npy_path = features_dir / f"sample_{idx}.npy"
        np.save(str(npy_path), data)
        npy_files.append(npy_path)
        
    # Write feature_index.csv
    index_path = tmp_path / "feature_index.csv"
    with open(index_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_path", "original_audio_path", "processed_audio_path", "dataset", "speaker_id", "emotion", "feature_type", "feature_shape"])
        for idx, emotion in enumerate(emotions):
            writer.writerow([
                str(npy_files[idx]),          # feature_path
                f"raw_audio_{idx}.wav",        # original_audio_path
                f"proc_audio_{idx}.wav",       # processed_audio_path
                "RAVDESS",                     # dataset
                f"actor_{idx:02d}",            # speaker_id
                emotion,                       # emotion
                "mel",                         # feature_type
                "(128, 94)"                    # feature_shape
            ])
            
    return index_path, features_dir, npy_files


def test_dataset_loading_and_len(mock_dataset_environment):
    """Tests successful instantiation, parsing, and length retrieval."""
    index_path, features_dir, _ = mock_dataset_environment
    
    config = VoiceDatasetConfig(
        index_path=index_path,
        features_dir=features_dir,
        emotions=["happy", "sad", "neutral"]
    )
    
    dataset = EmotionDataset(config=config)
    assert len(dataset) == 3


def test_dataset_getitem(mock_dataset_environment):
    """Tests loading, tensor conversion, shape verification, and label encoding."""
    index_path, features_dir, _ = mock_dataset_environment
    
    config = VoiceDatasetConfig(
        index_path=index_path,
        features_dir=features_dir,
        emotions=["happy", "sad", "neutral"]
    )
    
    dataset = EmotionDataset(config=config)
    
    # Sample index 0: happy (class 0)
    tensor_0, label_0 = dataset[0]
    assert isinstance(tensor_0, torch.Tensor)
    assert tensor_0.shape == (1, 128, 94)
    assert label_0 == 0
    # Values should all be 0.0 as generated in the fixture
    assert torch.all(tensor_0 == 0.0)
    
    # Sample index 1: sad (class 1)
    tensor_1, label_1 = dataset[1]
    assert tensor_1.shape == (1, 128, 94)
    assert label_1 == 1
    assert torch.all(tensor_1 == 1.0)


def test_dataset_invalid_label_in_index(tmp_path):
    """Tests that Dataset raises InvalidLabelError when indexing an unsupported emotion."""
    index_path = tmp_path / "feature_index.csv"
    with open(index_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["feature_path", "emotion"])
        writer.writerow(["sample.npy", "excited"])  # 'excited' is not standard
        
    config = VoiceDatasetConfig(
        index_path=index_path,
        features_dir=tmp_path
    )
    
    # Should raise error immediately during index scan
    with pytest.raises(InvalidLabelError):
        EmotionDataset(config=config)


def test_dataset_missing_index_file(tmp_path):
    """Tests exception when index file does not exist."""
    non_existent = tmp_path / "missing.csv"
    config = VoiceDatasetConfig(index_path=non_existent, features_dir=tmp_path)
    
    with pytest.raises(DatasetIOError):
        EmotionDataset(config=config)


def test_dataset_missing_feature_file(mock_dataset_environment):
    """Tests that Dataset raises DatasetIOError if a feature file goes missing."""
    index_path, features_dir, npy_files = mock_dataset_environment
    
    config = VoiceDatasetConfig(
        index_path=index_path,
        features_dir=features_dir,
        emotions=["happy", "sad", "neutral"]
    )
    
    dataset = EmotionDataset(config=config)
    
    # Delete one of the npy files
    npy_files[1].unlink()
    
    # Getting index 0 should succeed
    _ = dataset[0]
    
    # Getting index 1 should raise DatasetIOError
    with pytest.raises(DatasetIOError):
        _ = dataset[1]


def test_dataset_invalid_feature_dimensions(mock_dataset_environment):
    """Tests that Dataset raises DatasetIOError if feature file shape is incorrect."""
    index_path, features_dir, npy_files = mock_dataset_environment
    
    config = VoiceDatasetConfig(
        index_path=index_path,
        features_dir=features_dir,
        emotions=["happy", "sad", "neutral"]
    )
    
    dataset = EmotionDataset(config=config)
    
    # Overwrite sample_1 with invalid dimensions (e.g. 1D or wrong 2D size)
    invalid_data = np.zeros((128, 90), dtype=np.float32)  # width is 90 instead of 94
    np.save(str(npy_files[1]), invalid_data)
    
    with pytest.raises(DatasetIOError):
        _ = dataset[1]

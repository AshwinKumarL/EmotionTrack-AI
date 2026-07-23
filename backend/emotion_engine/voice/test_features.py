import pytest
import numpy as np
import soundfile as sf
import csv
import json
import shutil
from pathlib import Path

from backend.emotion_engine.voice.features import (
    MelSpectrogramExtractor,
    FeatureExtractorInitError,
    FeatureComputationError,
    FeatureIOError,
    FeatureDatasetBuilder
)

@pytest.fixture
def temp_audio_file(tmp_path):
    """
    Generate a 3-second synthetic sine wave audio file (16 kHz, mono) for testing.
    """
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 440 Hz tone
    audio_data = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    
    audio_file = tmp_path / "synthetic_440hz.wav"
    sf.write(str(audio_file), audio_data, sr)
    return audio_file, sr


def test_extractor_initialization():
    # 1. Invalid sample rate
    with pytest.raises(FeatureExtractorInitError):
        MelSpectrogramExtractor(sample_rate=-16000)
        
    # 2. Invalid n_fft
    with pytest.raises(FeatureExtractorInitError):
        MelSpectrogramExtractor(n_fft=0)
        
    # 3. Invalid hop_length
    with pytest.raises(FeatureExtractorInitError):
        MelSpectrogramExtractor(hop_length=-1)
        
    # 4. Invalid n_mels
    with pytest.raises(FeatureExtractorInitError):
        MelSpectrogramExtractor(n_mels=0)


def test_mel_spectrogram_extraction(temp_audio_file):
    audio_file, sr = temp_audio_file
    
    n_mels = 128
    hop_length = 512
    n_fft = 1024
    
    extractor = MelSpectrogramExtractor(
        sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    # Run extract
    features = extractor.extract(audio_file)
    
    # Assert type and shape
    assert isinstance(features, np.ndarray)
    assert features.ndim == 2
    assert features.shape[0] == n_mels
    
    # Expected frames formula with center padding (default in librosa.feature.melspectrogram is True)
    # Total samples = 3.0 * 16000 = 48000
    # Expected frames = 48000 // 512 + 1 = 94 frames
    expected_frames = int((sr * 3.0) // hop_length) + 1
    assert features.shape[1] == expected_frames
    
    # Check that values are decibel compressed (e.g. max is 0.0 or close, since ref=np.max)
    assert np.max(features) == pytest.approx(0.0, abs=1e-5)


def test_extractor_missing_file():
    extractor = MelSpectrogramExtractor()
    non_existent = Path("non_existent_file.wav")
    
    with pytest.raises(FeatureComputationError):
        extractor.extract(non_existent)


def test_save_and_load_features(tmp_path, temp_audio_file):
    audio_file, sr = temp_audio_file
    extractor = MelSpectrogramExtractor(sample_rate=sr)
    
    features = extractor.extract(audio_file)
    save_path = tmp_path / "features.npy"
    
    # Test saving
    extractor.save(features, save_path)
    assert save_path.exists()
    
    # Test loading
    loaded_features = extractor.load(save_path)
    assert np.array_equal(features, loaded_features)


def test_load_non_existent_feature_file():
    extractor = MelSpectrogramExtractor()
    non_existent = Path("non_existent_features.npy")
    
    with pytest.raises(FeatureIOError):
        extractor.load(non_existent)


def test_builder_runs_correctly(tmp_path, temp_audio_file):
    audio_file, sr = temp_audio_file
    
    # 1. Create a dummy dataset directory structure
    raw_dir = tmp_path / "datasets" / "raw"
    processed_dir = tmp_path / "datasets" / "processed"
    features_dir = tmp_path / "datasets" / "features"
    metadata_dir = tmp_path / "datasets" / "metadata"
    
    metadata_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy synthetic audio file to simulate RAVDESS dataset structure
    raw_ravdess_file = raw_dir / "RAVDESS" / "Actor_01" / "03-01-01-01-01-01-01.wav"
    raw_ravdess_file.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(raw_ravdess_file), np.ones(1000), sr) # dummy raw file
    
    # Processed file that builder actually processes
    processed_ravdess_file = processed_dir / "RAVDESS" / "Actor_01" / "03-01-01-01-01-01-01.wav"
    processed_ravdess_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(str(audio_file), str(processed_ravdess_file))
    
    # Add a file in raw index that is missing in processed to test skipped counting
    raw_missing_file = raw_dir / "CREMA-D" / "1001_DFA_ANG_XX.wav"
    raw_missing_file.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(raw_missing_file), np.ones(1000), sr) # dummy raw file
    
    # 2. Write a dummy dataset_index.csv
    index_csv = metadata_dir / "dataset_index.csv"
    with open(index_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file_path", "dataset_name", "emotion", "speaker_id"])
        # File 1: Success case
        writer.writerow([raw_ravdess_file.as_posix(), "RAVDESS", "neutral", "01"])
        # File 2: Skipped case (processed file not created)
        writer.writerow([raw_missing_file.as_posix(), "CREMA-D", "anger", "1001"])
        
    # 3. Setup extractor & builder
    extractor = MelSpectrogramExtractor(sample_rate=sr, n_mels=128)
    builder = FeatureDatasetBuilder(
        index_path=index_csv,
        processed_dir=processed_dir,
        features_dir=features_dir,
        extractor=extractor
    )
    
    # Run the builder
    stats = builder.run()
    
    # 4. Verify results
    assert stats["files_processed"] == 1
    assert stats["files_skipped"] == 1
    assert stats["extraction_failures"] == 0
    assert stats["feature_dimensions"]
    assert "(" in stats["feature_dimensions"] and ")" in stats["feature_dimensions"]
    
    # Verify created directories & files
    feature_meta_dir = features_dir / "metadata"
    assert feature_meta_dir.exists()
    assert (feature_meta_dir / "feature_config.json").exists()
    assert (feature_meta_dir / "feature_index.csv").exists()
    assert (feature_meta_dir / "feature_report.csv").exists()
    
    # Verify generated npy file exists
    expected_npy = features_dir / "mel" / "RAVDESS" / "Actor_01" / "03-01-01-01-01-01-01.npy"
    assert expected_npy.exists()
    
    # Load and assert features
    loaded_data = extractor.load(expected_npy)
    assert loaded_data.shape[0] == 128
    
    # Verify metadata contents
    with open(feature_meta_dir / "feature_index.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["dataset"] == "RAVDESS"
        assert rows[0]["emotion"] == "neutral"
        assert rows[0]["speaker_id"] == "01"
        assert rows[0]["feature_type"] == "mel"
        assert rows[0]["feature_shape"] == str(loaded_data.shape)
        
    # Verify config contents
    with open(feature_meta_dir / "feature_config.json", "r", encoding="utf-8") as f:
        config = json.load(f)
        assert config["feature_type"] == "mel"
        assert config["sample_rate"] == sr
        assert config["n_mels"] == 128

    # Verify report contents
    with open(feature_meta_dir / "feature_report.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        report_rows = list(reader)
        assert len(report_rows) == 2  # 1 success + 1 skipped
        # Row 1 (success)
        assert report_rows[0]["processed_audio_path"] == processed_ravdess_file.as_posix()
        assert report_rows[0]["feature_path"] == expected_npy.as_posix()
        assert report_rows[0]["status"] == "success"
        assert report_rows[0]["error_message"] == ""
        # Row 2 (skipped)
        assert report_rows[1]["processed_audio_path"] == (processed_dir / "CREMA-D" / "1001_DFA_ANG_XX.wav").as_posix()
        assert report_rows[1]["feature_path"] == ""
        assert report_rows[1]["status"] == "skipped"
        assert "does not exist" in report_rows[1]["error_message"]

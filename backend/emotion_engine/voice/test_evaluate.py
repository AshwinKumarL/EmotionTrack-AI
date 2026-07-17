import csv
import json
import numpy as np
import pytest
import torch
from pathlib import Path

from backend.emotion_engine.voice.dataset import VoiceDatasetConfig, EmotionDataset
from backend.emotion_engine.voice.model import VoiceModelConfig, EmotionCNN
from backend.emotion_engine.voice.evaluate import (
    EvaluationConfig,
    EvaluationError,
    ModelLoadError,
    DatasetLoadError,
    EvaluationRuntimeError,
    load_trained_model,
    generate_confusion_matrix_plot,
    run_evaluation
)


@pytest.fixture
def mock_evaluation_environment(tmp_path):
    """
    Creates a mock environment for evaluation testing containing:
    - 5 synthetic feature files
    - A mock feature_index.csv file
    - A mock model weight state dict file
    - An output directory path
    """
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    
    # Create 5 synthetic files (representing 2 classes: happy, sad)
    npy_files = []
    emotions = ["happy", "sad", "neutral", "calm", "happy"]
    
    for idx, emotion in enumerate(emotions):
        data = np.zeros((128, 94), dtype=np.float32)
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
            
    # Recreate weight state dict
    model_config = VoiceModelConfig(
        num_classes=8,
        input_channels=1,
        dropout_rate=0.5,
        filter_sizes=(32, 64, 128),
        kernel_sizes=(3, 3, 3),
        pool_sizes=(2, 2, 2),
        hidden_size=256
    )
    model = EmotionCNN(model_config)
    model_save_path = tmp_path / "best_model.pth"
    torch.save(model.state_dict(), str(model_save_path))
    
    output_dir = tmp_path / "reports"
    
    return model_save_path, index_path, features_dir, output_dir


def test_evaluation_config_validation(tmp_path):
    """Verify that EvaluationConfig handles bounds checking correctly."""
    valid_path = tmp_path / "best_model.pth"
    
    # 1. Invalid model path type
    with pytest.raises(EvaluationError):
        EvaluationConfig(
            model_path="string_instead_of_path",
            index_path=valid_path,
            features_dir=tmp_path,
            output_dir=tmp_path
        )
        
    # 2. Invalid validation split bounds
    with pytest.raises(EvaluationError):
        EvaluationConfig(
            model_path=valid_path,
            index_path=valid_path,
            features_dir=tmp_path,
            output_dir=tmp_path,
            validation_split=1.2  # must be between 0 and 1
        )


def test_model_loading_missing_file(tmp_path):
    """Verify that load_trained_model raises ModelLoadError if checkpoint file is missing."""
    non_existent = tmp_path / "missing_model.pth"
    with pytest.raises(ModelLoadError):
        load_trained_model(non_existent, 8, torch.device("cpu"))


def test_model_loading_corrupted_file(tmp_path):
    """Verify that load_trained_model raises ModelLoadError if file is not a valid checkpoint."""
    corrupted_path = tmp_path / "corrupted.pth"
    with open(corrupted_path, "w") as f:
        f.write("not a pytorch model checkpoint file")
        
    with pytest.raises(ModelLoadError):
        load_trained_model(corrupted_path, 8, torch.device("cpu"))


def test_dataset_loading_error(tmp_path):
    """Verify that run_evaluation handles missing files by raising DatasetLoadError."""
    non_existent_index = tmp_path / "missing_index.csv"
    config = EvaluationConfig(
        model_path=tmp_path / "best_model.pth",
        index_path=non_existent_index,
        features_dir=tmp_path,
        output_dir=tmp_path
    )
    
    with pytest.raises(DatasetLoadError):
        run_evaluation(config)


def test_confusion_matrix_generation(tmp_path):
    """Verify that confusion matrix plot is correctly saved to disk."""
    y_true = np.array([0, 1, 0, 1, 2])
    y_pred = np.array([0, 1, 1, 1, 2])
    labels = ["neutral", "calm", "happy"]
    
    save_path = tmp_path / "cm.png"
    generate_confusion_matrix_plot(y_true, y_pred, labels, save_path)
    
    assert save_path.exists(), "Confusion matrix plot file was not created."
    assert save_path.stat().st_size > 0


def test_evaluation_pipeline_success(mock_evaluation_environment):
    """Verify that the evaluation pipeline completes successfully and writes reports."""
    model_save_path, index_path, features_dir, output_dir = mock_evaluation_environment
    
    config = EvaluationConfig(
        model_path=model_save_path,
        index_path=index_path,
        features_dir=features_dir,
        output_dir=output_dir,
        validation_split=0.4,  # split 2 of 5 samples for validation, 3 for training
        random_seed=42,
        device="cpu"
    )
    
    metrics = run_evaluation(config)
    
    # Assert metric fields
    assert "accuracy" in metrics
    assert "precision_macro" in metrics
    assert "recall_macro" in metrics
    assert "f1_macro" in metrics
    
    # Verify outputs exist in the reports folder
    metrics_json = output_dir / "evaluation_metrics.json"
    report_txt = output_dir / "evaluation_report.txt"
    cm_png = output_dir / "confusion_matrix.png"
    
    assert metrics_json.exists()
    assert report_txt.exists()
    assert cm_png.exists()
    
    # Read metrics to verify structure
    with open(metrics_json, "r", encoding="utf-8") as f:
        saved_metrics = json.load(f)
        
    assert "accuracy" in saved_metrics
    assert isinstance(saved_metrics["accuracy"], float)

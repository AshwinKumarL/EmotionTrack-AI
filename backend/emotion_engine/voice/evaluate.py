import csv
import os
import sys
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# Setup path manipulation so the script can be run directly from any directory
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.emotion_engine.voice.loader import VoiceEngineError
from backend.emotion_engine.voice.dataset import VoiceDatasetConfig, EmotionDataset, LabelEncoder
from backend.emotion_engine.voice.model import VoiceModelConfig, EmotionCNN
from backend.emotion_engine.voice.train import VoiceTrainingConfig, setup_data_loaders

# =====================================================================
# Exception Hierarchy
# =====================================================================

class EvaluationError(VoiceEngineError):
    """
    Base exception class for all errors arising within the Evaluation pipeline.
    Inherits from VoiceEngineError.
    """
    pass


class ModelLoadError(EvaluationError):
    """
    Exception raised when loading model architecture or checkpoint state dict fails.
    """
    pass


class DatasetLoadError(EvaluationError):
    """
    Exception raised when loading dataset index, features, or loaders fails.
    """
    pass


class EvaluationRuntimeError(EvaluationError):
    """
    Exception raised during evaluation inference execution.
    """
    pass


# =====================================================================
# Configuration Class
# =====================================================================

@dataclass(frozen=True)
class EvaluationConfig:
    """
    Immutable configuration settings for model evaluation.
    """
    model_path: Path
    index_path: Path
    features_dir: Path
    output_dir: Path
    validation_split: float = 0.2
    random_seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self) -> None:
        """
        Validates configuration attributes.
        """
        if not isinstance(self.model_path, Path):
            raise EvaluationError("model_path must be a pathlib.Path object.")
        if not isinstance(self.index_path, Path):
            raise EvaluationError("index_path must be a pathlib.Path object.")
        if not isinstance(self.features_dir, Path):
            raise EvaluationError("features_dir must be a pathlib.Path object.")
        if not isinstance(self.output_dir, Path):
            raise EvaluationError("output_dir must be a pathlib.Path object.")
        if not (0.0 < self.validation_split < 1.0):
            raise EvaluationError("validation_split must satisfy the range 0.0 < split < 1.0.")


# =====================================================================
# Model Weight Loader
# =====================================================================

def load_trained_model(
    model_path: Path,
    num_classes: int,
    device: torch.device
) -> EmotionCNN:
    """
    Recreates the CNN architecture and loads the saved model state dict weights.
    
    Args:
        model_path (Path): Path to the saved weights file (.pth).
        num_classes (int): Number of target emotion categories.
        device (torch.device): Device to run the model on.
        
    Returns:
        EmotionCNN: The instantiated and loaded model in eval mode.
        
    Raises:
        ModelLoadError: If loading model configs or weights fails.
    """
    if not model_path.exists():
        raise ModelLoadError(f"Trained model checkpoint not found at: '{model_path}'")
        
    try:
        # Recreate model configuration (must match training parameters)
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
        
        # Load weights
        state_dict = torch.load(str(model_path), map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    except Exception as e:
        raise ModelLoadError(f"Failed to load model from path '{model_path}': {e}") from e


# =====================================================================
# Plotting Helpers
# =====================================================================

def generate_confusion_matrix_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: List[str],
    save_path: Path
) -> None:
    """
    Generates and saves a labeled confusion matrix plot using Matplotlib.
    
    Args:
        y_true (np.ndarray): Target ground truth classes.
        y_pred (np.ndarray): Predicted classes.
        labels (List[str]): List of class names.
        save_path (Path): Destination save path.
        
    Raises:
        EvaluationRuntimeError: If plotting fails.
    """
    try:
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        im = plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title("Emotion Classification Confusion Matrix")
        plt.colorbar(im)
        
        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=45)
        plt.yticks(tick_marks, labels)
        
        # Annotate counts inside the confusion matrix cells
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(
                    j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black"
                )
                
        plt.ylabel("True Emotion Label")
        plt.xlabel("Predicted Emotion Label")
        plt.tight_layout()
        
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(str(save_path), bbox_inches='tight', dpi=150)
        plt.close()
    except Exception as e:
        raise EvaluationRuntimeError(f"Failed to generate confusion matrix plot: {e}") from e


# =====================================================================
# Evaluation Logic Orchestrator
# =====================================================================

def run_evaluation(config: EvaluationConfig) -> Dict[str, Any]:
    """
    Executes model evaluation over the validation/test partition and saves metrics.
    
    Args:
        config (EvaluationConfig): Evaluation settings.
        
    Returns:
        Dict[str, Any]: Mapping of metric names to float scores.
        
    Raises:
        EvaluationError: If model/dataset loading or evaluation runs fail.
    """
    device = torch.device(config.device)

    # 1. Dataset Load
    if not config.index_path.exists():
        raise DatasetLoadError(f"Dataset index file not found at: '{config.index_path}'")
        
    try:
        dataset_config = VoiceDatasetConfig(
            index_path=config.index_path,
            features_dir=config.features_dir
        )
        dataset = EmotionDataset(config=dataset_config)
    except Exception as e:
        raise DatasetLoadError(f"Failed to load dataset: {e}") from e

    # 2. Reconstruct Model and load weights
    num_classes = dataset.label_encoder.num_classes()
    model = load_trained_model(config.model_path, num_classes, device)

    # 3. Partition Data and Setup DataLoader
    # Recreate the exact same random splitting used during model training
    try:
        mock_train_config = VoiceTrainingConfig(
            batch_size=32,
            epochs=1,
            learning_rate=0.001,
            weight_decay=0.0001,
            validation_split=config.validation_split,
            random_seed=config.random_seed,
            device=config.device,
            num_workers=0,
            model_save_path=config.model_path,
            patience=7
        )
        _, val_loader = setup_data_loaders(dataset, mock_train_config)
    except Exception as e:
        raise DatasetLoadError(f"Failed to split dataset and setup loaders: {e}") from e

    # 4. Inference loop
    all_preds = []
    all_targets = []
    
    try:
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                logits = model(features)
                preds = torch.argmax(logits, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(labels.numpy())
    except Exception as e:
        raise EvaluationRuntimeError(f"Error during validation inference forward loop: {e}") from e

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    # 5. Compute Metrics
    # Rationale for Macro Averaging: Macro averaging computes the metric independently for 
    # each class and then takes the average. This treats all emotion classes equally, 
    # which is appropriate here as it ensures less-frequent emotions (like disgust or surprises)
    # have equal weight in final scores as common emotions, protecting against class-imbalance bias.
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    emotions_list = dataset.label_encoder.emotions
    report_txt = classification_report(
        y_true, y_pred, labels=list(range(len(emotions_list))), target_names=emotions_list, zero_division=0
    )

    metrics_dict = {
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1)
    }

    # Print results to stdout
    print("\n==================================================")
    print("Voice Model Evaluation Results")
    print("--------------------------------------------------")
    print(f"Overall Accuracy : {acc * 100:.2f}%")
    print(f"Precision (Macro): {prec * 100:.2f}%")
    print(f"Recall (Macro)   : {rec * 100:.2f}%")
    print(f"F1 Score (Macro) : {f1 * 100:.2f}%")
    print("\nClassification Report:")
    print(report_txt)
    print("==================================================")

    # 6. Save Report and Metrics
    try:
        config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON Metrics
        json_path = config.output_dir / "evaluation_metrics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metrics_dict, f, indent=4)
            
        # Save TXT Report
        report_path = config.output_dir / "evaluation_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("Voice Emotion Recognition Engine - Evaluation Report\n")
            f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("==================================================\n")
            f.write(f"Overall Accuracy : {acc * 100:.2f}%\n")
            f.write(f"Precision (Macro): {prec * 100:.2f}%\n")
            f.write(f"Recall (Macro)   : {rec * 100:.2f}%\n")
            f.write(f"F1 Score (Macro) : {f1 * 100:.2f}%\n")
            f.write("==================================================\n\n")
            f.write("Classification Report:\n")
            f.write(report_txt)
            
        # Save Confusion Matrix Plot
        cm_path = config.output_dir / "confusion_matrix.png"
        generate_confusion_matrix_plot(y_true, y_pred, emotions_list, cm_path)
        
        print(f"Metrics JSON saved to: {json_path}")
        print(f"Report text saved to: {report_path}")
        print(f"Confusion Matrix plot saved to: {cm_path}")
        
    except Exception as e:
        raise EvaluationRuntimeError(f"Failed to write evaluation reports to disk: {e}") from e

    return metrics_dict


# =====================================================================
# CLI Entry Point
# =====================================================================

if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent.parent.parent.parent
    
    # Configure path references
    model_save = root_dir / "models" / "best_voice_model.pth"
    index_csv = root_dir / "datasets" / "features" / "metadata" / "feature_index.csv"
    features_dir = root_dir / "datasets" / "features"
    reports_dir = root_dir / "reports"

    try:
        eval_config = EvaluationConfig(
            model_path=model_save,
            index_path=index_csv,
            features_dir=features_dir,
            output_dir=reports_dir
        )
        run_evaluation(eval_config)
    except Exception as e:
        print(f"Evaluation Failed: {e}", file=sys.stderr)
        sys.exit(1)

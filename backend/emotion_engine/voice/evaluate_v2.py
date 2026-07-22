"""
Voice Model Version 2 (VM/2) — Evaluation Script

Evaluates the trained VM/2 model (5-class) and generates:
  - Overall metrics (accuracy, macro precision, macro recall, macro F1)
  - Per-class metrics JSON (precision, recall, F1, support per emotion)
  - Classification report text
  - Confusion matrix plot

All outputs are saved to reports/v2/, completely separate from VM/1.

This script imports shared components from the existing VM/1 modules
(dataset.py, model.py, evaluate.py, train.py) to avoid code duplication.
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    precision_recall_fscore_support
)

# Setup path manipulation so the script can be run directly from any directory
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.emotion_engine.voice.dataset import VoiceDatasetConfig, EmotionDataset
from backend.emotion_engine.voice.model import VoiceModelConfig, EmotionCNN
from backend.emotion_engine.voice.train import VoiceTrainingConfig, setup_data_loaders
from backend.emotion_engine.voice.evaluate import (
    ModelLoadError,
    DatasetLoadError,
    EvaluationRuntimeError,
    generate_confusion_matrix_plot
)
from backend.emotion_engine.voice.train_v2 import VM2_EMOTIONS

# =====================================================================
# VM/2 Model Loader
# =====================================================================

def load_v2_model(
    model_path: Path,
    device: torch.device
) -> EmotionCNN:
    """
    Loads the VM/2 model from a metadata-rich checkpoint.

    Unlike VM/1's bare state_dict loading, VM/2 checkpoints contain
    model_config, emotions, epoch, and optimizer state alongside the
    model weights.

    Args:
        model_path (Path): Path to best_voice_model_v2.pth.
        device (torch.device): Target device.

    Returns:
        EmotionCNN: Loaded model in eval mode.

    Raises:
        ModelLoadError: If loading fails.
    """
    if not model_path.exists():
        raise ModelLoadError(f"VM/2 checkpoint not found at: '{model_path}'")

    try:
        checkpoint = torch.load(str(model_path), map_location=device)

        # Extract model config from checkpoint metadata
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

        # Print checkpoint metadata
        print(f"VM/2 model loaded from: {model_path}")
        print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Best val loss:    {checkpoint.get('best_val_loss', 'N/A')}")
        print(f"  Emotions:         {checkpoint.get('emotions', 'N/A')}")

        return model

    except KeyError as e:
        raise ModelLoadError(
            f"VM/2 checkpoint at '{model_path}' is missing expected key: {e}. "
            f"Ensure this is a VM/2 metadata-rich checkpoint, not a VM/1 bare state_dict."
        ) from e
    except Exception as e:
        raise ModelLoadError(f"Failed to load VM/2 model from '{model_path}': {e}") from e


# =====================================================================
# VM/2 Evaluation Orchestrator
# =====================================================================

def run_v2_evaluation(
    model_path: Path,
    index_path: Path,
    features_dir: Path,
    output_dir: Path,
    validation_split: float = 0.2,
    random_seed: int = 42,
    device_str: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Dict[str, Any]:
    """
    Executes VM/2 model evaluation on the validation partition.

    Args:
        model_path (Path): Path to the VM/2 checkpoint.
        index_path (Path): Path to feature_index_v2.csv.
        features_dir (Path): Base features directory.
        output_dir (Path): Directory to save evaluation outputs (reports/v2/).
        validation_split (float): Fraction used for validation (must match training).
        random_seed (int): Random seed (must match training for same split).
        device_str (str): Device string ('cuda' or 'cpu').

    Returns:
        Dict[str, Any]: Overall metrics dictionary.

    Raises:
        EvaluationError: If any step fails.
    """
    device = torch.device(device_str)

    # 1. Load Dataset (5-class)
    if not index_path.exists():
        raise DatasetLoadError(f"VM/2 index file not found at: '{index_path}'")

    try:
        dataset_config = VoiceDatasetConfig(
            index_path=index_path,
            features_dir=features_dir,
            emotions=VM2_EMOTIONS
        )
        dataset = EmotionDataset(config=dataset_config)
        print(f"VM/2 dataset loaded: {len(dataset)} samples, {len(VM2_EMOTIONS)} classes")
    except Exception as e:
        raise DatasetLoadError(f"Failed to load VM/2 dataset: {e}") from e

    # 2. Load Model
    model = load_v2_model(model_path, device)

    # 3. Setup DataLoader (same split as training for consistent evaluation)
    try:
        mock_train_config = VoiceTrainingConfig(
            batch_size=32,
            epochs=1,
            learning_rate=0.001,
            weight_decay=0.0001,
            validation_split=validation_split,
            random_seed=random_seed,
            device=device_str,
            num_workers=0,
            model_save_path=model_path,
            patience=7
        )
        _, val_loader = setup_data_loaders(dataset, mock_train_config)
    except Exception as e:
        raise DatasetLoadError(f"Failed to setup data loaders: {e}") from e

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
        raise EvaluationRuntimeError(f"Error during VM/2 inference loop: {e}") from e

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    # 5. Compute Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    emotions_list = dataset.label_encoder.emotions
    report_txt = classification_report(
        y_true, y_pred,
        labels=list(range(len(emotions_list))),
        target_names=emotions_list,
        zero_division=0
    )

    # Per-class metrics
    per_prec, per_rec, per_f1, per_support = precision_recall_fscore_support(
        y_true, y_pred,
        labels=list(range(len(emotions_list))),
        zero_division=0
    )

    overall_metrics = {
        "accuracy": float(acc),
        "precision_macro": float(prec),
        "recall_macro": float(rec),
        "f1_macro": float(f1)
    }

    per_class_metrics = {}
    for idx, emotion in enumerate(emotions_list):
        per_class_metrics[emotion] = {
            "precision": float(per_prec[idx]),
            "recall": float(per_rec[idx]),
            "f1": float(per_f1[idx]),
            "support": int(per_support[idx])
        }

    # Print results
    print("\n==================================================")
    print("Voice Model V2 Evaluation Results")
    print("--------------------------------------------------")
    print(f"Overall Accuracy : {acc * 100:.2f}%")
    print(f"Precision (Macro): {prec * 100:.2f}%")
    print(f"Recall (Macro)   : {rec * 100:.2f}%")
    print(f"F1 Score (Macro) : {f1 * 100:.2f}%")
    print("\nClassification Report:")
    print(report_txt)
    print("==================================================")

    # 6. Save Reports
    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Overall metrics JSON
        json_path = output_dir / "evaluation_metrics.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(overall_metrics, f, indent=4)

        # Per-class metrics JSON
        per_class_path = output_dir / "per_class_metrics.json"
        with open(per_class_path, "w", encoding="utf-8") as f:
            json.dump(per_class_metrics, f, indent=4)

        # Text report
        report_path = output_dir / "evaluation_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("Voice Model V2 — Evaluation Report\n")
            f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Emotions: {', '.join(VM2_EMOTIONS)}\n")
            f.write("==================================================\n")
            f.write(f"Overall Accuracy : {acc * 100:.2f}%\n")
            f.write(f"Precision (Macro): {prec * 100:.2f}%\n")
            f.write(f"Recall (Macro)   : {rec * 100:.2f}%\n")
            f.write(f"F1 Score (Macro) : {f1 * 100:.2f}%\n")
            f.write("==================================================\n\n")
            f.write("Classification Report:\n")
            f.write(report_txt)

        # Confusion matrix plot (reused from VM/1 evaluate.py)
        cm_path = output_dir / "confusion_matrix.png"
        generate_confusion_matrix_plot(y_true, y_pred, emotions_list, cm_path)

        print(f"\nMetrics JSON saved to:      {json_path}")
        print(f"Per-class metrics saved to: {per_class_path}")
        print(f"Report text saved to:       {report_path}")
        print(f"Confusion matrix saved to:  {cm_path}")

    except Exception as e:
        raise EvaluationRuntimeError(f"Failed to write VM/2 evaluation reports: {e}") from e

    return overall_metrics


# =====================================================================
# CLI Entry Point
# =====================================================================

if __name__ == "__main__":
    root_dir = Path(__file__).resolve().parent.parent.parent.parent

    model_save = root_dir / "models" / "best_voice_model_v2.pth"
    index_csv = root_dir / "datasets" / "features" / "metadata" / "feature_index_v2.csv"
    features_dir = root_dir / "datasets" / "features"
    reports_dir = root_dir / "reports" / "v2"

    try:
        run_v2_evaluation(
            model_path=model_save,
            index_path=index_csv,
            features_dir=features_dir,
            output_dir=reports_dir
        )
    except Exception as e:
        print(f"VM/2 Evaluation Failed: {e}", file=sys.stderr)
        sys.exit(1)

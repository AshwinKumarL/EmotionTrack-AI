"""
Voice Model Version 2 (VM/2) — Training Script

A 5-class conversational emotion recognition model that removes neutral,
surprised, and disgust from the training pipeline.

VM/2 Emotions: happy, sad, angry, fearful, calm

Training improvements over VM/1:
  - Weighted CrossEntropyLoss (compensates for calm class imbalance)
  - ReduceLROnPlateau learning rate scheduler
  - Metadata-rich checkpoint saving
  - Training log with epoch tracking (best_epoch, early_stop_epoch)

This script imports and reuses shared infrastructure from the existing
VM/1 modules (dataset.py, model.py, train.py) — no code duplication.
"""

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import random_split, Subset
from tqdm import tqdm

# Setup path manipulation so the script can be run directly from any directory
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.emotion_engine.voice.dataset import VoiceDatasetConfig, EmotionDataset
from backend.emotion_engine.voice.model import VoiceModelConfig, EmotionCNN
from backend.emotion_engine.voice.train import VoiceTrainingConfig, setup_data_loaders, TrainingError

# =====================================================================
# VM/2 Constants
# =====================================================================

VM2_EMOTIONS = ["happy", "sad", "angry", "fearful", "calm"]


# =====================================================================
# Filtered Dataset Index Generator
# =====================================================================

def generate_v2_index_if_needed(
    source_index: Path,
    target_index: Path,
    allowed_emotions: List[str]
) -> int:
    """
    Creates a filtered copy of the VM/1 feature index, keeping only rows
    whose emotion is in the allowed list. Skips generation if the target
    file already exists on disk.

    Args:
        source_index (Path): Path to the original VM/1 feature_index.csv.
        target_index (Path): Path to write feature_index_v2.csv.
        allowed_emotions (List[str]): Emotions to retain.

    Returns:
        int: Number of rows written (or -1 if file already existed).

    Raises:
        TrainingError: If the source index is missing or I/O fails.
    """
    if target_index.exists():
        print(f"VM/2 index already exists at: {target_index} (skipping regeneration)")
        return -1

    if not source_index.exists():
        raise TrainingError(f"VM/1 source index not found at: {source_index}")

    allowed_set = {e.strip().lower() for e in allowed_emotions}

    try:
        with open(source_index, mode="r", encoding="utf-8") as src:
            reader = csv.DictReader(src)
            fieldnames = reader.fieldnames

            rows = [row for row in reader if row.get("emotion", "").strip().lower() in allowed_set]

        target_index.parent.mkdir(parents=True, exist_ok=True)
        with open(target_index, mode="w", newline="", encoding="utf-8") as dst:
            writer = csv.DictWriter(dst, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"VM/2 index generated: {len(rows)} samples written to {target_index}")
        return len(rows)

    except Exception as e:
        raise TrainingError(f"Failed to generate VM/2 index: {e}") from e


# =====================================================================
# Class Weight Computation
# =====================================================================

def compute_class_weights(
    dataset: EmotionDataset,
    train_subset: Subset,
    num_classes: int,
    device: torch.device
) -> torch.Tensor:
    """
    Computes inverse-frequency class weights from the training subset.

    Weight formula: weight[c] = total_train_samples / (num_classes * count[c])

    Args:
        dataset (EmotionDataset): The full dataset (for accessing samples).
        train_subset (Subset): The training partition (after split).
        num_classes (int): Number of emotion classes.
        device (torch.device): Target device for the weight tensor.

    Returns:
        torch.Tensor: FloatTensor of shape (num_classes,) with per-class weights.
    """
    class_counts = Counter()
    for idx in train_subset.indices:
        emotion = dataset.samples[idx]["emotion"]
        label = dataset.label_encoder.encode(emotion)
        class_counts[label] += 1

    total = sum(class_counts.values())
    weights = torch.zeros(num_classes, dtype=torch.float32)
    for cls_idx in range(num_classes):
        count = class_counts.get(cls_idx, 1)  # Avoid division by zero
        weights[cls_idx] = total / (num_classes * count)

    print(f"Class weights computed from {total} training samples:")
    for cls_idx in range(num_classes):
        emotion_name = dataset.label_encoder.decode(cls_idx)
        print(f"  {emotion_name:<10} count={class_counts.get(cls_idx, 0):>5}  weight={weights[cls_idx]:.4f}")

    return weights.to(device)


# =====================================================================
# VM/2 Training Orchestrator
# =====================================================================

def run_training_v2(
    model: EmotionCNN,
    dataset: EmotionDataset,
    config: VoiceTrainingConfig,
    log_dir: Path,
    smoke_test: bool = False
) -> Dict[str, object]:
    """
    Orchestrates the complete VM/2 training and validation pipeline.

    Differences from VM/1 run_training():
      - Weighted CrossEntropyLoss (computed from training subset frequencies)
      - ReduceLROnPlateau scheduler (monitors validation loss)
      - Metadata-rich checkpoint saving
      - Training log with best_epoch, early_stop_epoch tracking

    Args:
        model (EmotionCNN): The network architecture instance (num_classes=5).
        dataset (EmotionDataset): The loaded VM/2 dataset (5-class filtered).
        config (VoiceTrainingConfig): Training hyperparameter settings.
        log_dir (Path): Directory to save training_log.json.
        smoke_test (bool): If True, runs 1 epoch / 1 batch for sanity check.

    Returns:
        Dict[str, object]: History containing metrics, best_epoch, and early_stop info.
    """
    # 1. Setup Device
    device_name = config.device.lower()
    if "cuda" in device_name and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.", file=sys.stderr)
        device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    model.to(device)

    # 2. Setup Data Loaders (reuses VM/1 setup_data_loaders)
    train_loader, val_loader = setup_data_loaders(dataset, config)

    # Extract the train subset for class weight computation
    # setup_data_loaders uses the same seed, so we recreate the split to get indices
    total_len = len(dataset)
    val_len = int(total_len * config.validation_split)
    train_len = total_len - val_len
    generator = torch.Generator().manual_seed(config.random_seed)
    train_subset, _ = random_split(dataset, [train_len, val_len], generator=generator)

    # 3. Compute class weights and setup weighted loss
    num_classes = model.config.num_classes
    class_weights = compute_class_weights(dataset, train_subset, num_classes, device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 4. Setup Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # 5. Setup LR Scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6
    )

    # Track Metrics History
    history = {
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rates": [],
        "best_epoch": None,
        "early_stop_epoch": None,
        "total_epochs_run": 0
    }

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    epochs_to_run = 1 if smoke_test else config.epochs

    print(f"\nStarting VM/2 training on device: {device}")
    print(f"  Emotions: {VM2_EMOTIONS}")
    print(f"  Classes: {num_classes}")
    print(f"  Dataset size: {len(dataset)} (train={train_len}, val={val_len})")
    if smoke_test:
        print("[SMOKE TEST MODE ENABLED] Running 1 epoch and 1 batch only.")

    for epoch in range(1, epochs_to_run + 1):
        epoch_start_time = time.perf_counter()
        current_lr = optimizer.param_groups[0]["lr"]

        # ==========================================
        # TRAINING LOOP
        # ==========================================
        model.train()
        train_running_loss = 0.0
        train_correct = 0
        train_total = 0

        train_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{epochs_to_run} [Training]",
            unit="batch",
            leave=False
        )

        for batch_idx, (features, labels) in enumerate(train_bar):
            # Move tensors to configured hardware device
            features = features.to(device)
            labels = labels.to(device)

            # Optimization pass
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # Track running stats
            train_running_loss += loss.item() * features.size(0)
            _, predicted = torch.max(logits, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()

            # Update progress bar metrics
            current_loss = train_running_loss / train_total
            current_acc = (train_correct / train_total) * 100.0
            train_bar.set_postfix({
                "Loss": f"{current_loss:.4f}",
                "Acc": f"{current_acc:.2f}%"
            })

            # In smoke-test mode, break after the very first batch
            if smoke_test:
                break

        # Calculate epoch train averages
        epoch_train_loss = train_running_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100.0

        # ==========================================
        # VALIDATION LOOP
        # ==========================================
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0

        val_bar = tqdm(
            val_loader,
            desc=f"Epoch {epoch}/{epochs_to_run} [Validation]",
            unit="batch",
            leave=False
        )

        with torch.no_grad():
            for batch_idx, (features, labels) in enumerate(val_bar):
                features = features.to(device)
                labels = labels.to(device)

                logits = model(features)
                loss = criterion(logits, labels)

                val_running_loss += loss.item() * features.size(0)
                _, predicted = torch.max(logits, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                # Update progress bar metrics
                current_val_loss = val_running_loss / val_total
                current_val_acc = (val_correct / val_total) * 100.0
                val_bar.set_postfix({
                    "Loss": f"{current_val_loss:.4f}",
                    "Acc": f"{current_val_acc:.2f}%"
                })

                # In smoke-test mode, break after the very first batch
                if smoke_test:
                    break

        # Calculate epoch validation averages
        epoch_val_loss = val_running_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100.0

        # ==========================================
        # LR SCHEDULER STEP
        # ==========================================
        if not smoke_test:
            scheduler.step(epoch_val_loss)

        # ==========================================
        # MODEL CHECKPOINTING (metadata-rich)
        # ==========================================
        best_saved = False
        if not smoke_test:
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                best_epoch = epoch
                epochs_without_improvement = 0
                try:
                    config.model_save_path.parent.mkdir(parents=True, exist_ok=True)
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
                        "emotions": VM2_EMOTIONS,
                        "epoch": epoch,
                        "best_val_loss": float(best_val_loss),
                        "optimizer_state_dict": optimizer.state_dict()
                    }
                    torch.save(checkpoint, str(config.model_save_path))
                    best_saved = True
                except Exception as e:
                    print(f"Warning: Failed to save best model checkpoint: {e}", file=sys.stderr)
            else:
                epochs_without_improvement += 1
        else:
            # Simulate checkpoint save in smoke test outputs
            best_saved = True
            best_epoch = 1

        epoch_elapsed = time.perf_counter() - epoch_start_time

        # Update History
        history["epochs"].append(epoch)
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)
        history["learning_rates"].append(current_lr)

        # Print Epoch Summary
        print("==================================================")
        print(f"Epoch {epoch} Summary (VM/2)")
        print(f"Training Loss      : {epoch_train_loss:.4f}")
        print(f"Training Accuracy  : {epoch_train_acc:.2f}%")
        print(f"Validation Loss    : {epoch_val_loss:.4f}")
        print(f"Validation Accuracy: {epoch_val_acc:.2f}%")
        print(f"Learning Rate      : {current_lr:.6f}")
        print(f"Best Model Saved   : {'Yes' if best_saved else 'No'}")
        print(f"Epoch Time         : {epoch_elapsed:.2f}s")
        print("==================================================")

        # Early stopping
        if not smoke_test and epochs_without_improvement >= config.patience:
            history["early_stop_epoch"] = epoch
            print(f"\nEarly stopping triggered at epoch {epoch}: validation loss did not improve "
                  f"for {config.patience} consecutive epochs.")
            break

    # Finalize history
    history["best_epoch"] = best_epoch
    history["total_epochs_run"] = len(history["epochs"])
    if history["early_stop_epoch"] is None:
        history["early_stop_epoch"] = None  # Training completed all epochs

    # Save training log
    if not smoke_test:
        try:
            log_dir.mkdir(parents=True, exist_ok=True)
            log_path = log_dir / "training_log.json"
            with open(log_path, "w", encoding="utf-8") as f:
                json.dump(history, f, indent=4)
            print(f"\nTraining log saved to: {log_path}")
        except Exception as e:
            print(f"Warning: Failed to save training log: {e}", file=sys.stderr)

    return history


# =====================================================================
# Main CLI Entry Point
# =====================================================================

def parse_args() -> Tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(description="Voice Model V2 Training CLI (5-class)")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a minimal smoke test (1 epoch, 1 batch) to verify the VM/2 training pipeline."
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run the full VM/2 training pipeline."
    )
    return parser, parser.parse_args()


if __name__ == "__main__":
    parser, args = parse_args()

    if not args.smoke_test and not args.train:
        parser.print_help()
        sys.exit(0)

    # Configure path defaults relative to this file
    root_dir = Path(__file__).resolve().parent.parent.parent.parent
    v1_index_csv = root_dir / "datasets" / "features" / "metadata" / "feature_index.csv"
    v2_index_csv = root_dir / "datasets" / "features" / "metadata" / "feature_index_v2.csv"
    features_dir = root_dir / "datasets" / "features"
    model_save = root_dir / "models" / "best_voice_model_v2.pth"
    log_dir = root_dir / "logs" / "v2"

    try:
        # 1. Generate filtered index (only if it doesn't already exist)
        generate_v2_index_if_needed(v1_index_csv, v2_index_csv, VM2_EMOTIONS)

        # 2. Initialize dataset config with 5-class emotion list
        dataset_config = VoiceDatasetConfig(
            index_path=v2_index_csv,
            features_dir=features_dir,
            emotions=VM2_EMOTIONS
        )
        print("Loading VM/2 EmotionDataset...")
        dataset = EmotionDataset(config=dataset_config)
        print(f"Dataset loaded. Total samples: {len(dataset)}")

        # 3. Initialize model config (same architecture, 5 output classes)
        model_config = VoiceModelConfig(
            num_classes=len(VM2_EMOTIONS),
            input_channels=1,
            dropout_rate=0.5,
            filter_sizes=(32, 64, 128),
            kernel_sizes=(3, 3, 3),
            pool_sizes=(2, 2, 2),
            hidden_size=256
        )
        model = EmotionCNN(model_config)

        # 4. Initialize training config
        train_config = VoiceTrainingConfig(
            batch_size=32,
            epochs=45,
            learning_rate=0.001,
            weight_decay=0.0001,
            validation_split=0.2,
            random_seed=42,
            device="cuda" if torch.cuda.is_available() else "cpu",
            num_workers=0,
            model_save_path=model_save,
            patience=7
        )

        if args.smoke_test:
            run_training_v2(model, dataset, train_config, log_dir, smoke_test=True)
            print("\nVM/2 smoke test successfully completed.")
        elif args.train:
            print("Starting full VM/2 model training...")
            run_training_v2(model, dataset, train_config, log_dir, smoke_test=False)
            print("\nVM/2 model training successfully completed.")

    except Exception as e:
        print(f"Fatal Error during VM/2 training: {e}", file=sys.stderr)
        sys.exit(1)

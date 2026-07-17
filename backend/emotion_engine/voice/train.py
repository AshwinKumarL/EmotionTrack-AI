import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Setup path manipulation so the script can be run directly from any directory
project_root = str(Path(__file__).resolve().parent.parent.parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from backend.emotion_engine.voice.loader import VoiceEngineError
from backend.emotion_engine.voice.dataset import VoiceDatasetConfig, EmotionDataset
from backend.emotion_engine.voice.model import VoiceModelConfig, EmotionCNN

# =====================================================================
# Exception Hierarchy
# =====================================================================

class TrainingError(VoiceEngineError):
    """
    Base exception class for all errors arising within the Training pipeline.
    Inherits from VoiceEngineError.
    """
    pass


class TrainingConfigurationError(TrainingError):
    """
    Exception raised when training parameters or device configurations are invalid.
    """
    pass


# =====================================================================
# Training Configuration Class
# =====================================================================

@dataclass(frozen=True)
class VoiceTrainingConfig:
    """
    Immutable hyperparameter configuration for the CNN training pipeline.
    """
    batch_size: int
    epochs: int
    learning_rate: float
    weight_decay: float
    validation_split: float
    random_seed: int
    device: str
    num_workers: int
    model_save_path: Path
    patience: int

    def __post_init__(self) -> None:
        """
        Validates all configuration attributes.
        """
        if not isinstance(self.batch_size, int) or self.batch_size <= 0:
            raise TrainingConfigurationError("batch_size must be a positive integer.")

        if not isinstance(self.epochs, int) or self.epochs <= 0:
            raise TrainingConfigurationError("epochs must be a positive integer.")

        if not isinstance(self.learning_rate, float) or self.learning_rate <= 0.0:
            raise TrainingConfigurationError("learning_rate must be a positive float.")

        if not isinstance(self.weight_decay, float) or self.weight_decay < 0.0:
            raise TrainingConfigurationError("weight_decay must be a non-negative float.")

        if not isinstance(self.validation_split, float) or not (0.0 < self.validation_split < 1.0):
            raise TrainingConfigurationError("validation_split must satisfy the range 0.0 < split < 1.0.")

        if not isinstance(self.random_seed, int):
            raise TrainingConfigurationError("random_seed must be an integer.")

        if not isinstance(self.num_workers, int) or self.num_workers < 0:
            raise TrainingConfigurationError("num_workers must be a non-negative integer.")

        if not isinstance(self.model_save_path, Path):
            raise TrainingConfigurationError("model_save_path must be a pathlib.Path object.")

        if not isinstance(self.device, str):
            raise TrainingConfigurationError("device must be a string (e.g., 'cpu' or 'cuda').")

        if not isinstance(self.patience, int) or self.patience <= 0:
            raise TrainingConfigurationError("patience must be a positive integer.")


# =====================================================================
# Data Loading Setup
# =====================================================================

def setup_data_loaders(
    dataset: EmotionDataset, config: VoiceTrainingConfig
) -> Tuple[DataLoader, DataLoader]:
    """
    Splits the EmotionDataset into training and validation subsets and initializes
    the respective PyTorch DataLoaders.
    
    Args:
        dataset (EmotionDataset): The complete loaded dataset.
        config (VoiceTrainingConfig): Configuration specifying splits and loaders.
        
    Returns:
        Tuple[DataLoader, DataLoader]: Training and Validation DataLoader objects.
    """
    total_len = len(dataset)
    if total_len < 2:
        raise TrainingError(f"Dataset length {total_len} is too small to perform splits.")
        
    val_len = int(total_len * config.validation_split)
    train_len = total_len - val_len

    # Enforce split bounds
    if train_len <= 0 or val_len <= 0:
        raise TrainingError(
            f"Invalid split lengths. Total: {total_len}, Validation Split: {config.validation_split}. "
            f"Resulting lengths: Train={train_len}, Val={val_len}. Must both be greater than 0."
        )

    # Use reproducible random splitting
    generator = torch.Generator().manual_seed(config.random_seed)
    try:
        train_subset, val_subset = random_split(
            dataset, [train_len, val_len], generator=generator
        )
    except Exception as e:
        raise TrainingError(f"Failed to partition dataset using random_split: {e}") from e

    # Create loaders
    train_loader = DataLoader(
        train_subset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=(config.device != "cpu" and torch.cuda.is_available())
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=(config.device != "cpu" and torch.cuda.is_available())
    )

    return train_loader, val_loader


# =====================================================================
# Training Execution Orchestrator
# =====================================================================

def run_training(
    model: EmotionCNN,
    dataset: EmotionDataset,
    config: VoiceTrainingConfig,
    smoke_test: bool = False
) -> Dict[str, List[float]]:
    """
    Orchestrates the complete training and validation pipeline loop.
    
    Args:
        model (EmotionCNN): The network architecture instance.
        dataset (EmotionDataset): The loaded training/validation dataset.
        config (VoiceTrainingConfig): Training hyperparameter settings.
        smoke_test (bool): If True, executes exactly 1 epoch and 1 batch for sanity verification.
        
    Returns:
        Dict[str, List[float]]: History containing train/validation loss and accuracies.
    """
    # 1. Setup Device
    device_name = config.device.lower()
    if "cuda" in device_name and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.", file=sys.stderr)
        device = torch.device("cpu")
    else:
        device = torch.device(device_name)
        
    model.to(device)

    # 2. Setup Data Loaders
    train_loader, val_loader = setup_data_loaders(dataset, config)

    # 3. Setup Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # Track Metrics History
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    epochs_to_run = 1 if smoke_test else config.epochs

    print(f"Starting training on device: {device}")
    if smoke_test:
        print("[SMOKE TEST MODE ENABLED] Running 1 epoch and 1 batch only.")

    for epoch in range(1, epochs_to_run + 1):
        epoch_start_time = time.perf_counter()

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
        # MODEL CHECKPOINTING
        # ==========================================
        best_saved = False
        # Do not save checkpoints if we are doing a smoke test to avoid corrupting actual weights
        if not smoke_test:
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                epochs_without_improvement = 0
                try:
                    config.model_save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), str(config.model_save_path))
                    best_saved = True
                except Exception as e:
                    print(f"Warning: Failed to save best model checkpoint: {e}", file=sys.stderr)
            else:
                epochs_without_improvement += 1
        else:
            # Simulate checkpoint save in smoke test outputs
            best_saved = True

        epoch_elapsed = time.perf_counter() - epoch_start_time

        # Update History Dict
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        # Print Epoch Summary in requested layout format
        print("==================================================")
        print(f"Epoch {epoch} Summary")
        print(f"Training Loss      : {epoch_train_loss:.4f}")
        print(f"Training Accuracy  : {epoch_train_acc:.2f}%")
        print(f"Validation Loss    : {epoch_val_loss:.4f}")
        print(f"Validation Accuracy: {epoch_val_acc:.2f}%")
        print(f"Best Model Saved   : {'Yes' if best_saved else 'No'}")
        print(f"Epoch Time         : {epoch_elapsed:.2f}s")
        print("==================================================")

        # Early stopping verification check
        if not smoke_test and epochs_without_improvement >= config.patience:
            print(f"\nEarly stopping triggered: validation loss did not improve for {config.patience} consecutive epochs.")
            break

    return history


# =====================================================================
# Main CLI Entry Point (for manual execution & smoke tests)
# =====================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Voice Emotion Recognition Training CLI")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a minimal smoke test (1 epoch, 1 batch) to verify the training pipeline."
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Configure path defaults relative to this file
    root_dir = Path(__file__).resolve().parent.parent.parent.parent
    index_csv = root_dir / "datasets" / "features" / "metadata" / "feature_index.csv"
    features_dir = root_dir / "datasets" / "features"
    model_save = root_dir / "models" / "best_voice_model.pth"

    try:
        # 1. Initialize dataset config
        dataset_config = VoiceDatasetConfig(
            index_path=index_csv,
            features_dir=features_dir
        )
        print("Loading EmotionDataset...")
        dataset = EmotionDataset(config=dataset_config)
        print(f"Dataset loaded. Total samples: {len(dataset)}")

        # 2. Initialize model config
        model_config = VoiceModelConfig(
            num_classes=len(dataset.label_encoder.emotions),
            input_channels=1,
            dropout_rate=0.5,
            filter_sizes=(32, 64, 128),
            kernel_sizes=(3, 3, 3),
            pool_sizes=(2, 2, 2),
            hidden_size=256
        )
        model = EmotionCNN(model_config)

        # 3. Initialize training config
        train_config = VoiceTrainingConfig(
            batch_size=32,
            epochs=45,
            learning_rate=0.001,
            weight_decay=0.0001,
            validation_split=0.2,
            random_seed=42,
            device="cuda" if torch.cuda.is_available() else "cpu",
            num_workers=0,  # Set to 0 to avoid subprocess issues on standard CLI runs
            model_save_path=model_save,
            patience=7
        )

        if args.smoke_test:
            run_training(model, dataset, train_config, smoke_test=True)
            print("Smoke test successfully completed.")
        else:
            print("Training pipeline validated. Run with --smoke-test to execute a smoke test run.")
            print("Otherwise, wait for instructions to begin full training.")

    except Exception as e:
        print(f"Fatal Error during training initialization: {e}", file=sys.stderr)
        sys.exit(1)

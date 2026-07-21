# Voice Model Version 2 (VM/2) — Documentation

## Overview

VM/2 is a 5-class conversational emotion recognition model optimized for detecting core emotional states in spoken conversation. It runs alongside VM/1 as an independent model version.

## Differences from VM/1

| Aspect | VM/1 | VM/2 |
|--------|------|------|
| **Classes** | 8 emotions | 5 emotions |
| **Focus** | Full emotion spectrum | Conversational core emotions |
| **Loss Function** | `CrossEntropyLoss` | **Weighted** `CrossEntropyLoss` |
| **LR Scheduler** | None | `ReduceLROnPlateau` |
| **Checkpoint Format** | Bare `state_dict` | Metadata-rich dictionary |
| **Checkpoint Path** | `models/best_voice_model.pth` | `models/best_voice_model_v2.pth` |
| **Reports** | `reports/` | `reports/v2/` |
| **Training Log** | None | `logs/v2/training_log.json` |

## Removed Emotions

The following three emotions were removed from VM/2 to focus on conversational contexts:

| Emotion | Reason for Removal |
|---------|-------------------|
| **Neutral** | Low discriminative value in conversational emotion analysis; often a catch-all category that dilutes model confidence on expressive emotions. |
| **Surprised** | Acoustically ambiguous — often confused with happy/fearful. Only 384 samples available (RAVDESS-only), leading to unreliable training signal. |
| **Disgust** | Rare in natural conversation. Acoustically similar to angry, creating confusion in the classifier. |

## Label Mapping

VM/2 uses a 5-class label encoding:

| Index | Emotion |
|-------|---------|
| 0 | happy |
| 1 | sad |
| 2 | angry |
| 3 | fearful |
| 4 | calm |

## Dataset

VM/2 reuses the same preprocessed `.npy` Mel Spectrogram features as VM/1. A filtered index file (`feature_index_v2.csv`) selects only the 5 target emotions.

| Emotion | Samples | Proportion |
|---------|---------|------------|
| happy | 1,655 | 23.6% |
| sad | 1,655 | 23.6% |
| angry | 1,655 | 23.6% |
| fearful | 1,655 | 23.6% |
| calm | 384 | 5.5% |
| **Total** | **7,004** | **100%** |

> **Note:** The `calm` class has significantly fewer samples (~4.3× less than the majority classes). This imbalance is addressed by weighted CrossEntropyLoss.

## Training Improvements

### Weighted CrossEntropyLoss

Class weights are computed dynamically from the training subset using inverse-frequency scaling:

```
weight[c] = total_train_samples / (num_classes × class_count[c])
```

This ensures the `calm` class receives approximately 4.3× higher penalty for misclassification, compensating for its underrepresentation and improving recall on minority classes.

### ReduceLROnPlateau Scheduler

| Parameter | Value |
|-----------|-------|
| Mode | `min` (monitoring validation loss) |
| Factor | `0.5` (halves LR on plateau) |
| Patience | `3` (waits 3 epochs before reducing) |
| Min LR | `1e-6` (prevents LR from vanishing) |

The scheduler reduces the learning rate when validation loss plateaus, allowing the optimizer to fine-tune in later epochs. Combined with early stopping (patience=7), this creates a two-tier convergence strategy:
- **Epochs 1–3 of stagnation**: LR is halved, giving the model a chance to improve.
- **Epochs 4–6 of stagnation**: LR is halved again.
- **Epoch 7 of stagnation**: Early stopping triggers.

### Metadata-Rich Checkpoints

VM/2 checkpoints save a complete snapshot:

```python
{
    "model_state_dict": ...,      # Model weights
    "model_config": { ... },      # Architecture parameters
    "emotions": [...],            # Emotion list
    "epoch": 23,                  # Best epoch number
    "best_val_loss": 0.4821,      # Best validation loss
    "optimizer_state_dict": ...   # Optimizer state
}
```

This makes checkpoints self-describing and simplifies model loading, versioning, and debugging.

## Expected Benefits

1. **Better per-class accuracy** on happy, sad, and angry — removing acoustically confusing classes (neutral, disgust, surprised) reduces inter-class confusion.
2. **Improved calm recognition** — weighted loss compensates for the 4.3× imbalance, preventing the model from ignoring the minority class.
3. **More stable convergence** — the LR scheduler allows finer gradient steps as the model approaches convergence.
4. **Simpler decision boundary** — 5 classes vs 8 classes means a less complex classification problem for the same network capacity.

## File Inventory

| File | Purpose |
|------|---------|
| `backend/emotion_engine/voice/train_v2.py` | VM/2 training script |
| `backend/emotion_engine/voice/evaluate_v2.py` | VM/2 evaluation script |
| `backend/emotion_engine/voice/test_voice.py` | Inference tester (supports `--model VM1` / `--model VM2`) |
| `datasets/features/metadata/feature_index_v2.csv` | Filtered 5-class dataset index |
| `models/best_voice_model_v2.pth` | VM/2 trained checkpoint |
| `reports/v2/evaluation_metrics.json` | Overall evaluation metrics |
| `reports/v2/per_class_metrics.json` | Per-class precision, recall, F1, support |
| `reports/v2/evaluation_report.txt` | Full classification report |
| `reports/v2/confusion_matrix.png` | 5×5 confusion matrix plot |
| `logs/v2/training_log.json` | Training history with epoch tracking |
| `docs/voice_model_v2.md` | This documentation |

## Commands Reference

```bash
# Train VM/2 (full training run)
python backend/emotion_engine/voice/train_v2.py --train

# Smoke test VM/2 training pipeline
python backend/emotion_engine/voice/train_v2.py --smoke-test

# Evaluate VM/2
python backend/emotion_engine/voice/evaluate_v2.py

# Manual test with VM/2
python backend/emotion_engine/voice/test_voice.py --mic --model VM2
python backend/emotion_engine/voice/test_voice.py --file path/to/audio.wav --model VM2

# Manual test with VM/1 (default, unchanged)
python backend/emotion_engine/voice/test_voice.py --mic
python backend/emotion_engine/voice/test_voice.py --mic --model VM1
```

## Compatibility

- VM/1 remains fully functional. All VM/1 files, checkpoints, and reports are untouched.
- Both models can coexist in the same project without conflicts.
- `test_voice.py` defaults to VM/1 when `--model` is not specified.
- VM/2 reuses the existing `EmotionCNN`, `EmotionDataset`, `LabelEncoder`, and `VoiceDatasetConfig` from VM/1 modules — no code was duplicated.

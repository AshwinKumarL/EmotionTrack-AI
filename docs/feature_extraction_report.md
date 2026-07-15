# Feature Extraction Report

This report documents the execution of the Feature Extraction stage (Module 3) for the Voice Emotion Engine.

## Hyperparameter Configuration

- **Feature Type**: mel
- **Sample Rate**: 16000 Hz
- **n_fft**: 1024
- **hop_length**: 512
- **win_length**: 1024
- **Number of Mel Bins**: 128
- **Power**: 2.0
- **Decibel Scaling Limit (top_db)**: 80.0 dB

## Execution Statistics

- **Files processed successfully**: 10322
- **Files skipped (missing or error)**: 0
- **Extraction failures**: 0
- **Feature Dimensions**: (128, 94) (height x width / channels x time)
- **Total Execution Time**: 129.67 seconds

## Summary

The feature extraction run completed successfully. Standardized 2D log-Mel spectrogram feature matrices have been computed and saved as NumPy binary format (`.npy`) files in `datasets/features/mel/`. The feature dimensions are completely uniform, satisfying the requirements for mini-batch training with 2D Convolutional Neural Networks.

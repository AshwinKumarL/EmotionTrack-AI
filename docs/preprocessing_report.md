# Voice Preprocessing Pipeline Report

This report documents the verification and metrics of the offline preprocessing pipeline run.

## Preprocessing Run Configuration

- **Target Sample Rate**: 16000 Hz
- **Target Duration**: 3.0 seconds
- **Normalization Method**: Peak
- **Silence Trimming Enabled**: True
- **Silence Threshold**: 30.0 dB
- **Mono Conversion Enabled**: True

## Preprocessing Execution Statistics

- **Number of processed files**: 10322
- **Number of skipped files**: 0
- **Final sample rate**: 16000 Hz
- **Final clip duration**: 3.0 seconds
- **Number of stereo files converted**: 10
- **Number of files trimmed (silence)**: 5026
- **Number of files padded**: 9226
- **Number of files trimmed (duration)**: 1096
- **Total preprocessing time**: 143.18 seconds

## Summary

The preprocessing run successfully standardized 10322 files from RAVDESS and CREMA-D to a uniform sample rate of 16000 Hz and duration of 3.0 seconds.
One representative original vs. processed audio pair for each emotion has been saved in the `docs/preprocessing_examples/` directory.

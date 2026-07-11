# Dataset Setup and Verification Report

This report documents the verification, organization, and import of the audio datasets used for training the Voice Emotion Engine.

## Summary of Datasets

### RAVDESS (The Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Folder Location**: `datasets/raw/RAVDESS/`
- **Number of Audio Files**: 2,880
- **Number of Actor Folders**: 24 (Actors 01-24)
- **Number of Speakers**: 24 (12 male, 12 female)
- **Sample Filename**: `03-01-01-01-01-01-01.wav`
- **Sample Rate**: 48,000 Hz (48 kHz)
- **Emotion Labels Found**:
  - `neutral`
  - `calm`
  - `happy`
  - `sad`
  - `angry`
  - `fearful`
  - `disgust`
  - `surprised`

### CREMA-D (Crowd-sourced Emotional Multimodal Actors Dataset)
- **Folder Location**: `datasets/raw/CREMA-D/`
- **Number of Audio Files**: 7,442
- **Number of Speakers (Unique Speaker IDs)**: 91
- **Sample Filename**: `1001_DFA_ANG_XX.wav`
- **Sample Rate**: 16,000 Hz (16 kHz)
- **Emotion Labels Found**:
  - `neutral`
  - `happy`
  - `sad`
  - `angry`
  - `fearful`
  - `disgust`

---

## Integrity and Verification Details

- **Corrupted or Unreadable Files**: None. All 10,322 `.wav` files were successfully parsed and validated by opening a test slice via `librosa`.
- **CSV Index File**: Created at `datasets/metadata/dataset_index.csv`.
  - Structure:
    - `file_path`: Absolute path of the audio file.
    - `dataset_name`: Name of the dataset (`RAVDESS` or `CREMA-D`).
    - `emotion`: Lowercase standard emotion label.
    - `speaker_id`: Speaker/Actor identifier.

---

## Warnings or Issues
- **Sample Rate Discrepancy**: RAVDESS is recorded at a high quality of 48 kHz, while CREMA-D is recorded at 16 kHz. When training downstream models, downsampling RAVDESS to 16 kHz will be necessary for consistency.
- **Vocal Modality**: RAVDESS includes speech and song recordings. CREMA-D includes speech recordings only.
- **Filename Duplicate Paths in RAVDESS**: The original ZIP archive contained duplicate directories (`Actor_XX/` directly at the root, and another copy under `audio_speech_actors_01-24/Actor_XX/`). In accordance with the requirement to preserve original folder structures without deletes, both copies are preserved in the raw folder and indexed under the RAVDESS dataset.

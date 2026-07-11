import csv
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

from backend.emotion_engine.voice.loader import VoiceEngineError

# =====================================================================
# Custom Exceptions
# =====================================================================

class AudioPreprocessingError(VoiceEngineError):
    """
    Exception raised when a specific audio preprocessing step fails.
    """
    pass


# =====================================================================
# Normalizer Interface and Implementations
# =====================================================================

class AudioNormalizer(ABC):
    """
    Abstract base class defining the interface for audio amplitude normalizers.
    """
    @abstractmethod
    def normalize(self, waveform: np.ndarray) -> np.ndarray:
        """
        Normalize the given audio waveform.
        
        Args:
            waveform: 1D numpy array of audio amplitudes.
            
        Returns:
            Normalized 1D numpy array.
        """
        pass


class PeakNormalizer(AudioNormalizer):
    """
    Normalizes the waveform peak amplitude to a target value (default 1.0).
    Does not alter the dynamic range, only scales the signal linearly.
    """
    def __init__(self, target_peak: float = 1.0) -> None:
        self.target_peak = target_peak

    def normalize(self, waveform: np.ndarray) -> np.ndarray:
        if waveform.size == 0:
            return waveform
        max_val = np.max(np.abs(waveform))
        if max_val > 0:
            return waveform * (self.target_peak / max_val)
        return waveform


# =====================================================================
# Preprocessing Pipeline
# =====================================================================

class AudioPreprocessor:
    """
    Responsible for orchestrating the digital signal processing (DSP) pipeline
    for a single audio waveform.
    
    Steps executed in order:
    1. Stereo to mono conversion
    2. Resampling to a configurable sample rate
    3. Silence trimming
    4. Amplitude normalization
    5. Padding or trimming to a configurable duration
    """
    def __init__(
        self,
        target_sr: int = 16000,
        target_duration: float = 3.0,
        normalizer: Optional[AudioNormalizer] = None,
        silence_trimming_enabled: bool = True,
        silence_threshold_db: float = 30.0,
        mono_conversion_enabled: bool = True
    ) -> None:
        self.target_sr = target_sr
        self.target_duration = target_duration
        self.normalizer = normalizer or PeakNormalizer()
        self.silence_trimming_enabled = silence_trimming_enabled
        self.silence_threshold_db = silence_threshold_db
        self.mono_conversion_enabled = mono_conversion_enabled

    def convert_to_mono(self, waveform: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Converts stereo/multichannel waveform to mono by averaging channels.
        In librosa, stereo audio is expected to be of shape (channels, samples).
        """
        if not self.mono_conversion_enabled:
            return waveform, False

        if waveform.ndim > 1:
            # Check if actual multichannel or just single channel wrapped
            if waveform.shape[0] > 1:
                mono_wave = librosa.to_mono(waveform)
                return mono_wave, True
            else:
                # Shape is (1, samples), flatten to 1D
                return waveform.squeeze(0), False
        return waveform, False

    def resample_audio(self, waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """
        Resamples the waveform to target_sr using librosa's resample.
        """
        if orig_sr == target_sr:
            return waveform
        try:
            return librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)
        except Exception as e:
            raise AudioPreprocessingError(f"Resampling failed: {e}") from e

    def trim_silence(self, waveform: np.ndarray, top_db: float) -> Tuple[np.ndarray, bool]:
        """
        Trims leading and trailing silence using librosa.effects.trim.
        """
        if not self.silence_trimming_enabled:
            return waveform, False

        try:
            trimmed_wave, index = librosa.effects.trim(waveform, top_db=top_db)
            # If start > 0 or end < total samples, it was trimmed
            was_trimmed = (index[0] > 0) or (index[1] < len(waveform))
            return trimmed_wave, was_trimmed
        except Exception as e:
            raise AudioPreprocessingError(f"Silence trimming failed: {e}") from e

    def normalize_audio(self, waveform: np.ndarray) -> np.ndarray:
        """
        Normalizes amplitude of the waveform.
        """
        try:
            return self.normalizer.normalize(waveform)
        except Exception as e:
            raise AudioPreprocessingError(f"Normalization failed: {e}") from e

    def pad_or_trim_audio(self, waveform: np.ndarray, sr: int, target_duration: float) -> Tuple[np.ndarray, bool, bool]:
        """
        Pads or trims the audio clip to match target duration exactly.
        """
        target_samples = int(target_duration * sr)
        current_samples = len(waveform)

        if current_samples < target_samples:
            padding_len = target_samples - current_samples
            padded_wave = np.pad(waveform, (0, padding_len), mode="constant")
            return padded_wave, True, False
        elif current_samples > target_samples:
            trimmed_wave = waveform[:target_samples]
            return trimmed_wave, False, True
        else:
            return waveform, False, False

    def preprocess_audio(self, waveform: np.ndarray, orig_sr: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Applies the preprocessing pipeline steps in sequence.
        
        Returns:
            Tuple of:
            - Processed waveform (1D numpy array)
            - Metrics dictionary detailing what was performed
        """
        # Record original details (handle stereo/mono length consistently)
        original_duration = float(waveform.shape[-1] / orig_sr)

        # Step 1: Stereo to mono
        y, was_stereo_converted = self.convert_to_mono(waveform)

        # Step 2: Resample
        y = self.resample_audio(y, orig_sr, self.target_sr)

        # Step 3: Trim silence
        y, was_silence_trimmed = self.trim_silence(y, self.silence_threshold_db)

        # Step 4: Normalize
        y = self.normalize_audio(y)

        # Step 5: Pad/Trim
        y, was_padded, was_trimmed = self.pad_or_trim_audio(y, self.target_sr, self.target_duration)

        processed_duration = float(len(y) / self.target_sr)

        metrics = {
            "original_sample_rate": orig_sr,
            "processed_sample_rate": self.target_sr,
            "original_duration": original_duration,
            "processed_duration": processed_duration,
            "was_stereo_converted": was_stereo_converted,
            "was_silence_trimmed": was_silence_trimmed,
            "was_padding_applied": was_padded,
            "was_trimming_applied": was_trimmed,
        }
        return y, metrics


# =====================================================================
# Dataset Preprocessor Manager
# =====================================================================

class DatasetPreprocessor:
    """
    Orchestrates processing the whole dataset using paths from dataset_index.csv.
    Saves the processed outputs, generates config and reports.
    """
    def __init__(
        self,
        index_path: Path,
        processed_dir: Path,
        preprocessor: AudioPreprocessor,
        examples_dir: Optional[Path] = None
    ) -> None:
        self.index_path = index_path
        self.processed_dir = processed_dir
        self.preprocessor = preprocessor
        self.examples_dir = examples_dir or Path("docs/preprocessing_examples")
        
        # Setup logging
        self.logger = logging.getLogger("DatasetPreprocessor")
        
    def run(self) -> Dict[str, Any]:
        """
        Iterates through dataset_index.csv, runs the preprocessor,
        saves outputs, handles errors, and produces reports.
        
        Returns:
            A summary metrics dictionary.
        """
        if not self.index_path.exists():
            raise FileNotFoundError(f"Dataset index file not found at: {self.index_path}")

        # Create directories
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        self.examples_dir.mkdir(parents=True, exist_ok=True)

        # Track summary statistics
        stats = {
            "processed_count": 0,
            "skipped_count": 0,
            "stereo_converted_count": 0,
            "silence_trimmed_count": 0,
            "padded_count": 0,
            "trimmed_count": 0,
            "total_time": 0.0
        }

        start_time = time.time()

        # Store detail reports
        report_rows = []
        
        # To save one example per emotion
        saved_emotions = set()

        # Read the dataset index CSV
        with open(self.index_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.logger.info(f"Loaded {len(rows)} files from dataset index: {self.index_path}")

        # We use tqdm to display progress
        for row in tqdm(rows, desc="Preprocessing dataset"):
            raw_path_str = row["file_path"]
            dataset_name = row["dataset_name"]
            emotion = row["emotion"]
            speaker_id = row["speaker_id"]

            raw_path = Path(raw_path_str)

            # Compute processed target path preserving hierarchy
            normalized_path_parts = raw_path.parts
            try:
                # Find where "raw" lies under "datasets" to establish base folder layout
                raw_idx = -1
                for idx, part in enumerate(normalized_path_parts):
                    if part == "raw" and idx > 0 and normalized_path_parts[idx-1] == "datasets":
                        raw_idx = idx
                        break
                
                if raw_idx == -1:
                    # Fallback lookup
                    if "RAVDESS" in raw_path_str:
                        relative_parts = normalized_path_parts[normalized_path_parts.index("RAVDESS"):]
                    elif "CREMA-D" in raw_path_str:
                        relative_parts = normalized_path_parts[normalized_path_parts.index("CREMA-D"):]
                    else:
                        raise ValueError("Could not find datasets/raw or dataset folder in path")
                else:
                    relative_parts = normalized_path_parts[raw_idx + 1:]
                
                relative_path = Path(*relative_parts)
                processed_path = self.processed_dir / relative_path
            except Exception as e:
                self.logger.error(f"Failed to resolve output path for '{raw_path}': {e}")
                stats["skipped_count"] += 1
                report_rows.append({
                    "original_file_path": str(raw_path),
                    "processed_file_path": "",
                    "dataset": dataset_name,
                    "emotion": emotion,
                    "speaker": speaker_id,
                    "original_sample_rate": 0,
                    "processed_sample_rate": 0,
                    "original_duration": 0.0,
                    "processed_duration": 0.0,
                    "was_stereo_converted": False,
                    "was_silence_trimmed": False,
                    "was_padding_applied": False,
                    "was_trimming_applied": False,
                    "processing_status": f"Error: Path resolution failed ({e})"
                })
                continue

            # Load and process the file
            try:
                if not raw_path.exists():
                    raise FileNotFoundError(f"File does not exist: {raw_path}")

                # Load raw audio preserving original channels and sample rate
                waveform, orig_sr = librosa.load(str(raw_path), sr=None, mono=False)

                # Run the preprocessing pipeline
                processed_waveform, metrics = self.preprocessor.preprocess_audio(waveform, orig_sr)

                # Save the processed waveform
                processed_path.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(processed_path), processed_waveform, metrics["processed_sample_rate"])

                # Save learning examples (one per emotion)
                if emotion not in saved_emotions:
                    saved_emotions.add(emotion)
                    
                    orig_example_path = self.examples_dir / f"original_{emotion}.wav"
                    processed_example_path = self.examples_dir / f"processed_{emotion}.wav"
                    
                    # Save original (transpose if stereo for soundfile)
                    if waveform.ndim > 1:
                        sf.write(str(orig_example_path), waveform.T, orig_sr)
                    else:
                        sf.write(str(orig_example_path), waveform, orig_sr)
                        
                    sf.write(str(processed_example_path), processed_waveform, metrics["processed_sample_rate"])
                    self.logger.info(f"Saved learning example pair for emotion '{emotion}' to {self.examples_dir}")

                # Update statistics
                stats["processed_count"] += 1
                if metrics["was_stereo_converted"]:
                    stats["stereo_converted_count"] += 1
                if metrics["was_silence_trimmed"]:
                    stats["silence_trimmed_count"] += 1
                if metrics["was_padding_applied"]:
                    stats["padded_count"] += 1
                if metrics["was_trimming_applied"]:
                    stats["trimmed_count"] += 1

                # Add row to report
                report_rows.append({
                    "original_file_path": raw_path.as_posix(),
                    "processed_file_path": processed_path.as_posix(),
                    "dataset": dataset_name,
                    "emotion": emotion,
                    "speaker": speaker_id,
                    "original_sample_rate": orig_sr,
                    "processed_sample_rate": metrics["processed_sample_rate"],
                    "original_duration": round(metrics["original_duration"], 4),
                    "processed_duration": round(metrics["processed_duration"], 4),
                    "was_stereo_converted": metrics["was_stereo_converted"],
                    "was_silence_trimmed": metrics["was_silence_trimmed"],
                    "was_padding_applied": metrics["was_padding_applied"],
                    "was_trimming_applied": metrics["was_trimming_applied"],
                    "processing_status": "success"
                })

            except Exception as e:
                self.logger.error(f"Failed to process file '{raw_path}': {e}")
                stats["skipped_count"] += 1
                report_rows.append({
                    "original_file_path": raw_path.as_posix(),
                    "processed_file_path": "",
                    "dataset": dataset_name,
                    "emotion": emotion,
                    "speaker": speaker_id,
                    "original_sample_rate": 0,
                    "processed_sample_rate": 0,
                    "original_duration": 0.0,
                    "processed_duration": 0.0,
                    "was_stereo_converted": False,
                    "was_silence_trimmed": False,
                    "was_padding_applied": False,
                    "was_trimming_applied": False,
                    "processing_status": f"Failed: {e}"
                })

        # Calculate total execution time
        stats["total_time"] = time.time() - start_time
        self.logger.info(f"Completed preprocessing run. Processed: {stats['processed_count']}, Skipped: {stats['skipped_count']}.")

        # Save config JSON
        config_dir = self.processed_dir / "metadata"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "preprocessing_config.json"
        
        # Check normalizer class name
        norm_method = self.preprocessor.normalizer.__class__.__name__
        if isinstance(self.preprocessor.normalizer, PeakNormalizer):
            norm_method = "Peak"

        config_data = {
            "target_sample_rate": self.preprocessor.target_sr,
            "target_duration": self.preprocessor.target_duration,
            "normalization_method": norm_method,
            "silence_trimming_enabled": self.preprocessor.silence_trimming_enabled,
            "silence_threshold_db": self.preprocessor.silence_threshold_db,
            "mono_conversion_enabled": self.preprocessor.mono_conversion_enabled
        }
        with open(config_path, "w", encoding="utf-8") as cf:
            json.dump(config_data, cf, indent=4)
        self.logger.info(f"Saved preprocessing config to {config_path}")

        # Save report CSV
        csv_path = config_dir / "preprocessing_report.csv"
        csv_headers = [
            "original_file_path", "processed_file_path", "dataset", "emotion", "speaker",
            "original_sample_rate", "processed_sample_rate", "original_duration", "processed_duration",
            "was_stereo_converted", "was_silence_trimmed", "was_padding_applied", "was_trimming_applied",
            "processing_status"
        ]
        with open(csv_path, mode="w", newline="", encoding="utf-8") as cf:
            writer = csv.DictWriter(cf, fieldnames=csv_headers)
            writer.writeheader()
            writer.writerows(report_rows)
        self.logger.info(f"Saved preprocessing CSV report to {csv_path}")

        # Save Markdown Report
        report_md_path = self.examples_dir.parent / "preprocessing_report.md"
        report_md_content = f"""# Voice Preprocessing Pipeline Report

This report documents the verification and metrics of the offline preprocessing pipeline run.

## Preprocessing Run Configuration

- **Target Sample Rate**: {self.preprocessor.target_sr} Hz
- **Target Duration**: {self.preprocessor.target_duration} seconds
- **Normalization Method**: {norm_method}
- **Silence Trimming Enabled**: {self.preprocessor.silence_trimming_enabled}
- **Silence Threshold**: {self.preprocessor.silence_threshold_db} dB
- **Mono Conversion Enabled**: {self.preprocessor.mono_conversion_enabled}

## Preprocessing Execution Statistics

- **Number of processed files**: {stats['processed_count']}
- **Number of skipped files**: {stats['skipped_count']}
- **Final sample rate**: {self.preprocessor.target_sr} Hz
- **Final clip duration**: {self.preprocessor.target_duration} seconds
- **Number of stereo files converted**: {stats['stereo_converted_count']}
- **Number of files trimmed (silence)**: {stats['silence_trimmed_count']}
- **Number of files padded**: {stats['padded_count']}
- **Number of files trimmed (duration)**: {stats['trimmed_count']}
- **Total preprocessing time**: {stats['total_time']:.2f} seconds

## Summary

The preprocessing run successfully standardized {stats['processed_count']} files from RAVDESS and CREMA-D to a uniform sample rate of {self.preprocessor.target_sr} Hz and duration of {self.preprocessor.target_duration} seconds.
One representative original vs. processed audio pair for each emotion has been saved in the `docs/preprocessing_examples/` directory.
"""
        with open(report_md_path, "w", encoding="utf-8") as rmf:
            rmf.write(report_md_content)
        self.logger.info(f"Saved preprocessing MD report to {report_md_path}")

        return stats


# =====================================================================
# Main Execution Block
# =====================================================================

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )
    
    # Establish base directory
    base_dir = Path(__file__).resolve().parent.parent.parent.parent
    index_path = base_dir / "datasets" / "metadata" / "dataset_index.csv"
    processed_dir = base_dir / "datasets" / "processed"
    examples_dir = base_dir / "docs" / "preprocessing_examples"

    # Configure the preprocessor pipeline
    preprocessor = AudioPreprocessor(
        target_sr=16000,
        target_duration=3.0,
        silence_trimming_enabled=True,
        silence_threshold_db=30.0,
        mono_conversion_enabled=True
    )

    # Instantiate manager
    dataset_preprocessor = DatasetPreprocessor(
        index_path=index_path,
        processed_dir=processed_dir,
        preprocessor=preprocessor,
        examples_dir=examples_dir
    )

    logging.info("Starting offline Voice Emotion Engine preprocessing...")
    try:
        run_stats = dataset_preprocessor.run()
        logging.info("Offline preprocessing completed successfully!")
        logging.info(f"Summary -> Processed: {run_stats['processed_count']}, Skipped: {run_stats['skipped_count']}, Time: {run_stats['total_time']:.2f}s")
    except Exception as exc:
        logging.critical(f"Pipeline crashed: {exc}", exc_info=True)

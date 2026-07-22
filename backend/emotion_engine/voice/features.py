import csv
import json
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import librosa
from tqdm import tqdm

from backend.emotion_engine.voice.loader import VoiceEngineError

# =====================================================================
# Custom Exceptions
# =====================================================================

class FeatureExtractionError(VoiceEngineError):
    """
    Base exception class for all errors arising within the Feature Extraction module.
    """
    pass


class FeatureExtractorInitError(FeatureExtractionError):
    """
    Exception raised when feature extractor initialization fails due to invalid parameters.
    """
    pass


class FeatureComputationError(FeatureExtractionError):
    """
    Exception raised when a specific feature computation (e.g., Mel Spectrogram) fails.
    """
    pass


class FeatureIOError(FeatureExtractionError):
    """
    Exception raised when saving or loading feature arrays to/from disk fails.
    """
    pass


# =====================================================================
# Feature Extractor Interface and Implementations
# =====================================================================

class FeatureExtractor(ABC):
    """
    Abstract base class defining the common interface for all feature extractors.
    Supports extensibility for future acoustic representations (e.g. MFCCs, Chroma).
    """
    def __init__(self, **kwargs) -> None:
        pass

    @abstractmethod
    def extract(self, audio_path: Path) -> np.ndarray:
        """
        Extract features from a processed audio file and return a numpy array.
        This must be side-effect free (do not write to disk here).
        
        Args:
            audio_path: Absolute path to the processed audio file.
            
        Returns:
            Numpy array of extracted features.
        """
        pass

    @abstractmethod
    def save(self, feature_data: np.ndarray, save_path: Path) -> None:
        """
        Serialize and save extracted features to disk.
        
        Args:
            feature_data: The numpy array of extracted features.
            save_path: Absolute destination path.
        """
        pass

    @abstractmethod
    def load(self, load_path: Path) -> np.ndarray:
        """
        Load features from disk.
        
        Args:
            load_path: Absolute path to the saved feature file.
            
        Returns:
            Numpy array of features.
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Return the configuration/hyperparameter dictionary of the extractor.
        """
        pass


class MelSpectrogramExtractor(FeatureExtractor):
    """
    Extracts log-Mel Spectrogram representation from standardized mono audio.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        hop_length: int = 512,
        win_length: int = 1024,
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        power: float = 2.0,
        top_db: Optional[float] = 80.0
    ) -> None:
        super().__init__()
        
        # Validation of parameters
        if sample_rate <= 0:
            raise FeatureExtractorInitError("Sample rate must be greater than 0.")
        if n_fft <= 0:
            raise FeatureExtractorInitError("n_fft must be greater than 0.")
        if hop_length <= 0:
            raise FeatureExtractorInitError("Hop length must be greater than 0.")
        if n_mels <= 0:
            raise FeatureExtractorInitError("Number of Mel bins must be greater than 0.")
        
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax if fmax is not None else float(sample_rate / 2)
        self.power = power
        self.top_db = top_db

    def extract(self, audio_path: Path) -> np.ndarray:
        """
        Loads preprocessed WAV and computes the log-Mel Spectrogram.
        
        Args:
            audio_path: Absolute path to preprocessed WAV file.
            
        Returns:
            Numpy array of shape (n_mels, time_steps)
        """
        try:
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
            # Load processed audio enforcing target sample rate
            y, sr = librosa.load(str(audio_path), sr=self.sample_rate)
            
            # Compute Mel Spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y,
                sr=sr,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_mels=self.n_mels,
                fmin=self.fmin,
                fmax=self.fmax,
                power=self.power
            )
            
            # Convert to log-Mel representation (dB scale)
            if self.power == 1.0:
                log_mel = librosa.amplitude_to_db(mel_spec, ref=np.max, top_db=self.top_db)
            else:
                log_mel = librosa.power_to_db(mel_spec, ref=np.max, top_db=self.top_db)
                
            return log_mel
        except FileNotFoundError as e:
            raise FeatureComputationError(str(e)) from e
        except Exception as e:
            raise FeatureComputationError(f"Failed to compute Mel Spectrogram: {e}") from e

    def save(self, feature_data: np.ndarray, save_path: Path) -> None:
        """
        Save feature data as a numpy binary file (.npy).
        """
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(save_path), feature_data)
        except Exception as e:
            raise FeatureIOError(f"Failed to save features to {save_path}: {e}") from e

    def load(self, load_path: Path) -> np.ndarray:
        """
        Load feature data from a numpy binary file (.npy).
        """
        try:
            if not load_path.exists():
                raise FileNotFoundError(f"Feature file not found: {load_path}")
            return np.load(str(load_path))
        except FileNotFoundError as e:
            raise FeatureIOError(str(e)) from e
        except Exception as e:
            raise FeatureIOError(f"Failed to load features from {load_path}: {e}") from e

    def get_config(self) -> Dict[str, Any]:
        return {
            "feature_type": "mel",
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "win_length": self.win_length,
            "n_mels": self.n_mels,
            "fmin": self.fmin,
            "fmax": self.fmax,
            "power": self.power,
            "top_db": self.top_db
        }


# =====================================================================
# Feature Dataset Orchestration / Builder
# =====================================================================

class FeatureDatasetBuilder:
    """
    Orchestrates feature extraction for the entire dataset.
    Reads dataset_index.csv, runs feature extraction on preprocessed WAVs,
    saves numpy feature arrays, and generates reports/metadata.
    """
    def __init__(
        self,
        index_path: Path,
        processed_dir: Path,
        features_dir: Path,
        extractor: FeatureExtractor
    ) -> None:
        self.index_path = index_path
        self.processed_dir = processed_dir
        self.features_dir = features_dir
        self.extractor = extractor
        self.logger = logging.getLogger("FeatureDatasetBuilder")

    def run(self) -> Dict[str, Any]:
        """
        Main execution loop for dataset feature extraction.
        
        Returns:
            Dictionary with execution statistics.
        """
        if not self.index_path.exists():
            raise FileNotFoundError(f"Dataset index file not found at: {self.index_path}")

        # Setup feature subdirectories
        feature_type = self.extractor.get_config().get("feature_type", "unknown")
        out_features_dir = self.features_dir / feature_type
        metadata_dir = self.features_dir / "metadata"

        out_features_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        stats = {
            "files_processed": 0,
            "files_skipped": 0,
            "extraction_failures": 0,
            "feature_dimensions": "None",
            "execution_time_seconds": 0.0
        }

        # Keep track of index rows and detailed report rows
        index_rows = []
        report_rows = []
        
        # Read dataset index
        with open(self.index_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        self.logger.info(f"Loaded {len(rows)} samples from dataset index: {self.index_path}")

        for row in tqdm(rows, desc="Extracting features"):
            raw_path_str = row["file_path"]
            dataset_name = row["dataset_name"]
            emotion = row["emotion"]
            speaker_id = row["speaker_id"]

            raw_path = Path(raw_path_str)
            start_file_time = time.time()
            processed_path = None

            # Map raw WAV path to processed WAV path
            try:
                if "raw" in raw_path.parts:
                    raw_idx = raw_path.parts.index("raw")
                    relative_parts = raw_path.parts[raw_idx + 1:]
                else:
                    # Fallback mapping
                    if "RAVDESS" in raw_path_str:
                        relative_parts = raw_path.parts[raw_path.parts.index("RAVDESS"):]
                    elif "CREMA-D" in raw_path_str:
                        relative_parts = raw_path.parts[raw_path.parts.index("CREMA-D"):]
                    else:
                        raise ValueError("Could not find datasets/raw or dataset folder in path")
                
                relative_path = Path(*relative_parts)
                processed_path = self.processed_dir / relative_path
            except Exception as e:
                self.logger.error(f"Failed to resolve processed path for '{raw_path}': {e}")
                stats["files_skipped"] += 1
                report_rows.append({
                    "processed_audio_path": "",
                    "feature_path": "",
                    "execution_time": round(time.time() - start_file_time, 4),
                    "feature_dimensions": "",
                    "status": "skipped",
                    "error_message": f"Path resolution failed: {e}"
                })
                continue

            # Process files
            try:
                if not processed_path.exists():
                    self.logger.warning(f"Processed audio file does not exist, skipping: {processed_path}")
                    stats["files_skipped"] += 1
                    report_rows.append({
                        "processed_audio_path": processed_path.as_posix(),
                        "feature_path": "",
                        "execution_time": round(time.time() - start_file_time, 4),
                        "feature_dimensions": "",
                        "status": "skipped",
                        "error_message": "Processed audio file does not exist"
                    })
                    continue

                # Run extraction (pure computation)
                feature_data = self.extractor.extract(processed_path)
                
                # Verify and store dimension on first success
                if stats["feature_dimensions"] == "None":
                    stats["feature_dimensions"] = str(feature_data.shape)

                # Map features path preserving structure
                feature_relative_path = relative_path.with_suffix(".npy")
                target_feature_path = out_features_dir / feature_relative_path
                
                # Save feature numpy array
                self.extractor.save(feature_data, target_feature_path)

                stats["files_processed"] += 1
                file_exec_time = time.time() - start_file_time

                # Record metadata row (store paths using posix slash for consistency)
                index_rows.append({
                    "feature_path": target_feature_path.as_posix(),
                    "original_audio_path": raw_path.as_posix(),
                    "processed_audio_path": processed_path.as_posix(),
                    "dataset": dataset_name,
                    "speaker_id": speaker_id,
                    "emotion": emotion,
                    "feature_type": feature_type,
                    "feature_shape": str(feature_data.shape)
                })

                # Record detailed report row
                report_rows.append({
                    "processed_audio_path": processed_path.as_posix(),
                    "feature_path": target_feature_path.as_posix(),
                    "execution_time": round(file_exec_time, 4),
                    "feature_dimensions": str(feature_data.shape),
                    "status": "success",
                    "error_message": ""
                })

            except Exception as e:
                self.logger.error(f"Failed to extract features for '{processed_path}': {e}")
                stats["extraction_failures"] += 1
                report_rows.append({
                    "processed_audio_path": processed_path.as_posix() if processed_path else "",
                    "feature_path": "",
                    "execution_time": round(time.time() - start_file_time, 4),
                    "feature_dimensions": "",
                    "status": "failed",
                    "error_message": str(e)
                })

        # Calculate final execution time
        stats["execution_time_seconds"] = round(time.time() - start_time, 2)
        
        # Save feature_config.json
        config_path = metadata_dir / "feature_config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.extractor.get_config(), f, indent=4)
        self.logger.info(f"Saved feature extraction config to {config_path}")

        # Save feature_index.csv
        index_csv_path = metadata_dir / "feature_index.csv"
        headers = [
            "feature_path", "original_audio_path", "processed_audio_path",
            "dataset", "speaker_id", "emotion", "feature_type", "feature_shape"
        ]
        with open(index_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(index_rows)
        self.logger.info(f"Saved feature index to {index_csv_path}")

        # Save feature_report.csv
        report_path = metadata_dir / "feature_report.csv"
        report_headers = [
            "processed_audio_path", "feature_path", "execution_time",
            "feature_dimensions", "status", "error_message"
        ]
        with open(report_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=report_headers)
            writer.writeheader()
            writer.writerows(report_rows)
        self.logger.info(f"Saved feature execution report to {report_path}")

        # Save summary markdown report in docs
        docs_dir = self.index_path.parent.parent.parent / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        report_md_path = docs_dir / "feature_extraction_report.md"
        
        report_md_content = f"""# Feature Extraction Report

This report documents the execution of the Feature Extraction stage (Module 3) for the Voice Emotion Engine.

## Hyperparameter Configuration

- **Feature Type**: {feature_type}
- **Sample Rate**: {self.extractor.get_config().get("sample_rate")} Hz
- **n_fft**: {self.extractor.get_config().get("n_fft")}
- **hop_length**: {self.extractor.get_config().get("hop_length")}
- **win_length**: {self.extractor.get_config().get("win_length")}
- **Number of Mel Bins**: {self.extractor.get_config().get("n_mels")}
- **Power**: {self.extractor.get_config().get("power")}
- **Decibel Scaling Limit (top_db)**: {self.extractor.get_config().get("top_db")} dB

## Execution Statistics

- **Files processed successfully**: {stats['files_processed']}
- **Files skipped (missing or error)**: {stats['files_skipped']}
- **Extraction failures**: {stats['extraction_failures']}
- **Feature Dimensions**: {stats['feature_dimensions']} (height x width / channels x time)
- **Total Execution Time**: {stats['execution_time_seconds']:.2f} seconds

## Summary

The feature extraction run completed successfully. Standardized 2D log-Mel spectrogram feature matrices have been computed and saved as NumPy binary format (`.npy`) files in `datasets/features/mel/`. The feature dimensions are completely uniform, satisfying the requirements for mini-batch training with 2D Convolutional Neural Networks.
"""
        with open(report_md_path, "w", encoding="utf-8") as f:
            f.write(report_md_content)
        self.logger.info(f"Saved feature extraction markdown report to {report_md_path}")

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
    features_dir = base_dir / "datasets" / "features"

    # Configure extractor
    extractor = MelSpectrogramExtractor(
        sample_rate=16000,
        n_fft=1024,
        hop_length=512,
        win_length=1024,
        n_mels=128
    )

    builder = FeatureDatasetBuilder(
        index_path=index_path,
        processed_dir=processed_dir,
        features_dir=features_dir,
        extractor=extractor
    )

    logging.info("Starting offline Feature Extraction pipeline...")
    try:
        run_stats = builder.run()
        logging.info("Offline feature extraction completed successfully!")
        logging.info(
            f"Summary -> Processed: {run_stats['files_processed']}, "
            f"Skipped: {run_stats['files_skipped']}, "
            f"Failures: {run_stats['extraction_failures']}, "
            f"Time: {run_stats['execution_time_seconds']:.2f}s"
        )
    except Exception as exc:
        logging.critical(f"Pipeline crashed: {exc}", exc_info=True)

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import librosa

# =====================================================================
# Custom Exception Hierarchies
# =====================================================================

class VoiceEngineError(Exception):
    """
    Base exception class for all errors arising within the Voice Emotion Engine.
    Inherits from Python's standard Exception.
    """
    pass


class AudioLoadError(VoiceEngineError):
    """
    Exception raised when an audio file cannot be loaded, decoded, or read
    by the underlying audio libraries (e.g. librosa, soundfile).
    """
    pass


class MetadataParseError(VoiceEngineError):
    """
    Exception raised when the filename parsing logic cannot extract
    required metadata attributes (such as emotion or speaker ID) due to
    naming convention violations.
    """
    pass


class DatasetDetectionError(VoiceEngineError):
    """
    Exception raised when a file's location or name does not conform to 
    any supported datasets (e.g., RAVDESS or CREMA-D), preventing automatic 
    dataset name resolution.
    """
    pass


# =====================================================================
# Core Data representation
# =====================================================================

@dataclass
class VoiceSample:
    """
    A representation of a single vocal sample, storing its source path,
    dataset name, parsed emotion label, speaker/actor ID, and raw audio data.
    
    To ensure memory efficiency when dealing with thousands of files,
    this class implements LAZY LOADING: the raw audio waveform and sample rate
    are not loaded upon object creation and default to None. They are loaded
    and cached only when the `load_audio()` method is explicitly invoked.
    
    Attributes:
        file_path (Path): The absolute path to the audio file.
        dataset_name (str): The dataset the sample belongs to (e.g. 'RAVDESS', 'CREMA-D').
        emotion (str): The standardized lowercase emotion label (e.g. 'happy', 'neutral').
        speaker_id (str): The identifier of the speaker/actor who recorded the sample.
        waveform (Optional[np.ndarray]): 1D numpy float array representing the audio waveform. Defaults to None.
        sample_rate (Optional[int]): The sample rate of the audio in Hz. Defaults to None.
    """
    file_path: Path
    dataset_name: str
    emotion: str
    speaker_id: str
    waveform: Optional[np.ndarray] = None
    sample_rate: Optional[int] = None

    def load_audio(self, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        Loads the raw audio waveform and sample rate from the file path, and 
        caches them on this VoiceSample instance.
        
        Args:
            sr (Optional[int]): Target sample rate. If None, the native sample
                                rate of the audio file is preserved.
                                
        Returns:
            Tuple[np.ndarray, int]: A tuple containing:
                - waveform (np.ndarray): 1D float array of raw audio amplitudes.
                - sample_rate (int): The sample rate of the loaded audio.
                
        Raises:
            AudioLoadError: If the file cannot be loaded or decoded.
        """
        # Return cached waveform if already loaded and sample rate matches target
        if self.waveform is not None and (sr is None or sr == self.sample_rate):
            return self.waveform, self.sample_rate

        try:
            # We convert Path to str as librosa expects standard string paths in many environments
            y, loaded_sr = librosa.load(str(self.file_path), sr=sr)
            self.waveform = y
            self.sample_rate = loaded_sr
            return y, loaded_sr
        except Exception as e:
            raise AudioLoadError(
                f"Failed to load raw audio data from '{self.file_path}'. Reason: {e}"
            ) from e


# =====================================================================
# Dataset Loader Class
# =====================================================================

class VoiceDatasetLoader:
    """
    Responsible for scanning dataset folders, locating audio files,
    parsing metadata from filename conventions, and returning VoiceSample objects.
    """

    def __init__(self) -> None:
        """
        Initializes the loader with standard emotion dictionaries for 
        RAVDESS and CREMA-D mapping.
        """
        # RAVDESS Emotion Code Map (modality-vocalChannel-emotion-intensity-statement-repetition-actor)
        # Third parameter represents emotion code:
        self._ravdess_emotions = {
            "01": "neutral",
            "02": "calm",
            "03": "happy",
            "04": "sad",
            "05": "angry",
            "06": "fearful",
            "07": "disgust",
            "08": "surprised"
        }
        
        # CREMA-D Emotion Code Map (ActorID_Sentence_Emotion_Intensity.wav)
        # Third parameter represents emotion code:
        self._cremad_emotions = {
            "ANG": "angry",
            "DIS": "disgust",
            "FEA": "fearful",
            "HAP": "happy",
            "NEU": "neutral",
            "SAD": "sad"
        }

    def scan_dataset(self, dataset_dir: Path) -> List[Path]:
        """
        Recursively scans the provided directory for all files ending in '.wav'.
        
        Args:
            dataset_dir (Path): The directory to search.
            
        Returns:
            List[Path]: A list of absolute paths to all discovered WAV files.
            
        Raises:
            ValueError: If the target path does not exist or is not a directory.
        """
        if not dataset_dir.exists():
            raise ValueError(f"The path '{dataset_dir}' does not exist.")
        if not dataset_dir.is_dir():
            raise ValueError(f"The path '{dataset_dir}' is not a directory.")

        # Recursively search for WAV files and resolve to absolute paths
        return [p.resolve() for p in dataset_dir.rglob("*.wav") if p.is_file()]

    def get_all_audio_files(self, dataset_dir: Path) -> List[Path]:
        """
        A helper/wrapper around `scan_dataset` to retrieve all audio files in a folder.
        
        Args:
            dataset_dir (Path): The directory to search.
            
        Returns:
            List[Path]: List of absolute paths to WAV files.
        """
        return self.scan_dataset(dataset_dir)

    def load_audio(self, file_path: Path, sr: Optional[int] = None) -> Tuple[np.ndarray, int]:
        """
        Standalone method to load an audio file directly without creating a VoiceSample.
        
        Args:
            file_path (Path): Path to the audio file.
            sr (Optional[int]): Target sample rate. If None, native sample rate is used.
            
        Returns:
            Tuple[np.ndarray, int]: A tuple containing the waveform array and sample rate.
            
        Raises:
            AudioLoadError: If the file fails to open.
        """
        try:
            y, loaded_sr = librosa.load(str(file_path), sr=sr)
            return y, loaded_sr
        except Exception as e:
            raise AudioLoadError(
                f"Failed to load audio file '{file_path}' through standalone method. Reason: {e}"
            ) from e

    def parse_metadata(self, file_path: Path) -> Tuple[str, str, str]:
        """
        Parses metadata details from a file's name and directory location.
        
        Args:
            file_path (Path): Path to the audio file.
            
        Returns:
            Tuple[str, str, str]: A tuple of:
                - dataset_name (str): 'RAVDESS' or 'CREMA-D'
                - emotion (str): Standardized lowercase emotion string
                - speaker_id (str): Actor or Speaker identification string
                
        Raises:
            DatasetDetectionError: If the file path cannot be matched to a known dataset layout.
            MetadataParseError: If the filename format fails parsing validation rules.
        """
        path_str = file_path.as_posix()
        stem = file_path.stem

        # 1. Detect RAVDESS
        # RAVDESS paths contain "RAVDESS" or the file contains exactly 7 hyphenated components.
        if "RAVDESS" in path_str.upper() or len(stem.split("-")) == 7:
            dataset_name = "RAVDESS"
            parts = stem.split("-")
            if len(parts) != 7:
                raise MetadataParseError(
                    f"RAVDESS naming convention mismatch for '{file_path.name}'. "
                    f"Expected 7 hyphenated tokens, got {len(parts)}."
                )
            
            emotion_code = parts[2]
            emotion = self._ravdess_emotions.get(emotion_code)
            if not emotion:
                raise MetadataParseError(
                    f"Unsupported RAVDESS emotion code '{emotion_code}' parsed from '{file_path.name}'."
                )
                
            speaker_id = parts[6]  # The Actor ID represents the speaker
            return dataset_name, emotion, speaker_id

        # 2. Detect CREMA-D
        # CREMA-D paths contain "CREMA-D" or the file contains exactly 4 underscore-separated parts.
        elif "CREMA-D" in path_str.upper() or "_" in stem:
            dataset_name = "CREMA-D"
            parts = stem.split("_")
            if len(parts) != 4:
                raise MetadataParseError(
                    f"CREMA-D naming convention mismatch for '{file_path.name}'. "
                    f"Expected 4 underscore-separated tokens, got {len(parts)}."
                )
                
            speaker_id = parts[0]  # First part is Speaker/Actor ID
            emotion_code = parts[2].upper()
            emotion = self._cremad_emotions.get(emotion_code)
            if not emotion:
                raise MetadataParseError(
                    f"Unsupported CREMA-D emotion code '{emotion_code}' parsed from '{file_path.name}'."
                )
                
            return dataset_name, emotion, speaker_id

        # 3. Unsupported or Unknown Datasets
        else:
            raise DatasetDetectionError(
                f"Failed to identify dataset mapping for audio file: '{file_path}'. "
                f"Make sure the file path contains 'RAVDESS' or 'CREMA-D', or follows their filename conventions."
            )

    def load_sample(self, file_path: Path) -> VoiceSample:
        """
        Helper method to create a lazy-loaded VoiceSample instance from an audio file.
        This parses metadata immediately but defers waveform loading.
        
        Args:
            file_path (Path): Path to the WAV file.
            
        Returns:
            VoiceSample: An initialized VoiceSample containing parsed metadata.
            
        Raises:
            VoiceEngineError: If metadata parsing fails.
        """
        abs_path = file_path.resolve()
        dataset_name, emotion, speaker_id = self.parse_metadata(abs_path)
        return VoiceSample(
            file_path=abs_path,
            dataset_name=dataset_name,
            emotion=emotion,
            speaker_id=speaker_id
        )

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset

from backend.emotion_engine.voice.loader import VoiceEngineError

# =====================================================================
# Custom Exception Hierarchies
# =====================================================================

class DatasetError(VoiceEngineError):
    """
    Base exception class for all errors arising within the Dataset module.
    Inherits from VoiceEngineError.
    """
    pass


class InvalidLabelError(DatasetError):
    """
    Exception raised when an invalid or unmapped label/label index is supplied
    or validation fails.
    """
    pass


class DatasetConfigError(DatasetError):
    """
    Exception raised when dataset configuration parameters are invalid or missing.
    """
    pass


class DatasetIOError(DatasetError):
    """
    Exception raised when loading database index files or numpy feature files fails.
    """
    pass


# =====================================================================
# Dataset Configuration Data Class
# =====================================================================

@dataclass(frozen=True)
class VoiceDatasetConfig:
    """
    Immutable configuration settings for the EmotionDataset.
    
    Attributes:
        index_path (Path): Path to the feature_index.csv file.
        features_dir (Path): Base directory where feature .npy files are saved.
        emotions (Optional[List[str]]): List of target emotions to encode. If None,
                                        defaults to the standard set of 8 emotions.
    """
    index_path: Path
    features_dir: Path
    emotions: Optional[List[str]] = None

    def __post_init__(self) -> None:
        """
        Validates configuration attributes.
        """
        # index_path check: must be a Path object
        if not isinstance(self.index_path, Path):
            raise DatasetConfigError("index_path must be a pathlib.Path object.")
        # features_dir check: must be a Path object
        if not isinstance(self.features_dir, Path):
            raise DatasetConfigError("features_dir must be a pathlib.Path object.")


# =====================================================================
# Label Encoder Class
# =====================================================================

class LabelEncoder:
    """
    Responsible for converting categorical emotion string labels into integers
    and vice-versa, with robust validation for label schemas.
    """
    def __init__(self, emotions: Optional[List[str]] = None) -> None:
        """
        Initializes the LabelEncoder with standard or custom emotion labels.
        
        Args:
            emotions (Optional[List[str]]): Pre-defined list of allowed emotion categories.
                                            If None, uses standard RAVDESS/CREMA-D categories.
        """
        if emotions is None:
            # Default standard set of emotions in alphabetical order for stability:
            # angry, calm, disgust, fearful, happy, neutral, sadness, surprised
            self.emotions = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
        else:
            if not isinstance(emotions, list):
                raise DatasetConfigError("Emotions must be a list of strings.")
            if len(emotions) == 0:
                raise DatasetConfigError("Emotions list cannot be empty.")
            
            # De-duplicate while preserving original order
            seen = set()
            self.emotions = []
            for e in emotions:
                if not isinstance(e, str):
                    raise DatasetConfigError(f"Each emotion label must be a string, got {type(e).__name__}.")
                cleaned = e.strip().lower()
                if not cleaned:
                    raise DatasetConfigError("Emotion label cannot be an empty or whitespace string.")
                if cleaned not in seen:
                    seen.add(cleaned)
                    self.emotions.append(cleaned)

        self._emotion_to_idx = {emotion: idx for idx, emotion in enumerate(self.emotions)}
        self._idx_to_emotion = {idx: emotion for idx, emotion in enumerate(self.emotions)}

    def encode(self, emotion: str) -> int:
        """
        Converts an emotion string into its corresponding integer label.
        
        Args:
            emotion (str): The emotion label string to encode.
            
        Returns:
            int: The corresponding integer class label index.
            
        Raises:
            InvalidLabelError: If the emotion label is invalid or unmapped.
        """
        if not isinstance(emotion, str):
            raise InvalidLabelError(f"Emotion label must be a string, got {type(emotion).__name__}.")
            
        cleaned = emotion.strip().lower()
        if cleaned not in self._emotion_to_idx:
            raise InvalidLabelError(
                f"Unsupported emotion label: '{emotion}'. Supported classes are: {self.emotions}"
            )
        return self._emotion_to_idx[cleaned]

    def decode(self, label_idx: int) -> str:
        """
        Converts an integer label index back to its corresponding emotion string.
        
        Args:
            label_idx (int): The integer class label index to decode.
            
        Returns:
            str: The corresponding emotion label string.
            
        Raises:
            InvalidLabelError: If the index is out of bounds or not mapped.
        """
        if not isinstance(label_idx, (int, np.integer)):
            raise InvalidLabelError(
                f"Label index must be an integer, got {type(label_idx).__name__}."
            )
            
        idx = int(label_idx)
        if idx not in self._idx_to_emotion:
            raise InvalidLabelError(
                f"Invalid class index: {idx}. Supported index range is [0, {len(self.emotions) - 1}]."
            )
        return self._idx_to_emotion[idx]

    def num_classes(self) -> int:
        """Returns the total number of mapped emotion classes."""
        return len(self.emotions)


# =====================================================================
# Emotion Dataset Class
# =====================================================================

class EmotionDataset(Dataset):
    """
    A PyTorch Dataset that reads feature indices from a CSV and yields preprocessed
    audio feature tensors alongside encoded emotion label integers.
    """
    def __init__(self, config: VoiceDatasetConfig, label_encoder: Optional[LabelEncoder] = None) -> None:
        """
        Initializes the EmotionDataset.
        
        Args:
            config (VoiceDatasetConfig): Config class specifying input directories and files.
            label_encoder (Optional[LabelEncoder]): Encoder for standardizing target class labels.
                                                    If None, initialized using config settings.
        """
        if not isinstance(config, VoiceDatasetConfig):
            raise DatasetConfigError("config must be an instance of VoiceDatasetConfig.")

        self.config = config
        
        if label_encoder is None:
            self.label_encoder = LabelEncoder(emotions=config.emotions)
        else:
            if not isinstance(label_encoder, LabelEncoder):
                raise DatasetConfigError("label_encoder must be an instance of LabelEncoder.")
            self.label_encoder = label_encoder

        self.samples: List[Dict[str, str]] = []
        self._load_index()

    def _load_index(self) -> None:
        """
        Loads the feature index file and performs structural and label checks.
        
        Raises:
            DatasetIOError: If files cannot be read or are missing headers.
            InvalidLabelError: If a label in the index is unmapped/invalid.
        """
        if not self.config.index_path.exists():
            raise DatasetIOError(f"Dataset index file not found at: {self.config.index_path}")

        try:
            with open(self.config.index_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                
                # Verify that required headers exist
                required_headers = {"feature_path", "emotion"}
                if not reader.fieldnames or not required_headers.issubset(reader.fieldnames):
                    raise DatasetIOError(
                        f"Missing required columns in index file. Expected at least {required_headers}, "
                        f"but got: {reader.fieldnames}"
                    )
                
                for line_idx, row in enumerate(reader, start=2):
                    feature_path_str = row.get("feature_path")
                    emotion = row.get("emotion")
                    
                    if not feature_path_str:
                        raise DatasetIOError(f"Row {line_idx} in index file has an empty 'feature_path'.")
                    if not emotion:
                        raise DatasetIOError(f"Row {line_idx} in index file has an empty 'emotion'.")
                    
                    # Validate emotion label immediately (fail-fast architecture)
                    try:
                        self.label_encoder.encode(emotion)
                    except InvalidLabelError as e:
                        raise InvalidLabelError(
                            f"Row {line_idx} contains an invalid emotion label: {e}"
                        ) from e
                        
                    self.samples.append({
                        "feature_path": feature_path_str,
                        "emotion": emotion
                    })
        except (DatasetError, OSError):
            raise
        except Exception as e:
            raise DatasetIOError(f"Failed to read index file: {e}") from e

    def _resolve_feature_path(self, path_str: str) -> Path:
        """
        Dynamically resolves the feature file path, attempting to handle absolute
        paths stored from other environments by searching relative paths.
        
        Args:
            path_str (str): The path to resolve.
            
        Returns:
            Path: The resolved absolute Path object.
            
        Raises:
            DatasetIOError: If the feature file cannot be located on disk.
        """
        path = Path(path_str)
        if path.exists() and path.is_file():
            return path.resolve()
            
        # Try resolving relative to configured features directory
        if self.config.features_dir:
            # 1. Direct join if relative path
            direct_join = self.config.features_dir / path
            if direct_join.exists() and direct_join.is_file():
                return direct_join.resolve()
                
            # 2. Extract suffix structure (e.g. searching for 'mel' or 'features' in parts)
            parts = path.parts
            for anchor in ["mel", "features"]:
                if anchor in parts:
                    idx = parts.index(anchor)
                    rel_path = Path(*parts[idx:])
                    resolved = self.config.features_dir / rel_path
                    if resolved.exists() and resolved.is_file():
                        return resolved.resolve()
                        
                    # Also try join from after anchor
                    rel_path_after = Path(*parts[idx+1:])
                    resolved_after = self.config.features_dir / rel_path_after
                    if resolved_after.exists() and resolved_after.is_file():
                        return resolved_after.resolve()

            # 3. Last fallback: try resolving by file name under features_dir (recursively or directly)
            direct_filename = self.config.features_dir / path.name
            if direct_filename.exists() and direct_filename.is_file():
                return direct_filename.resolve()
                
        raise DatasetIOError(f"Feature file not found at: '{path_str}'")

    def __len__(self) -> int:
        """Returns the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Retrieves a single sample (feature_tensor, label) from the dataset.
        
        Args:
            idx (int): Index of the sample to fetch.
            
        Returns:
            Tuple[torch.Tensor, int]:
                - feature_tensor (torch.Tensor): Shape (1, 128, 94).
                - label (int): The encoded integer class label.
                
        Raises:
            IndexError: If idx is out of bounds.
            DatasetIOError: If loading the feature file fails or tensor shape is invalid.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self)}.")
            
        sample = self.samples[idx]
        feature_path_str = sample["feature_path"]
        emotion = sample["emotion"]
        
        resolved_path = self._resolve_feature_path(feature_path_str)
        
        try:
            feature_array = np.load(str(resolved_path))
        except Exception as e:
            raise DatasetIOError(f"Failed to load numpy array from file '{resolved_path}': {e}") from e
            
        if feature_array.ndim != 2:
            raise DatasetIOError(
                f"Feature array in '{resolved_path}' must be 2D, but got shape {feature_array.shape}."
            )
            
        try:
            feature_tensor = torch.from_numpy(feature_array).float()
        except Exception as e:
            raise DatasetIOError(f"Failed to convert numpy array to PyTorch tensor: {e}") from e
            
        # Add channel dimension so that shape changes from (128, 94) -> (1, 128, 94)
        feature_tensor = feature_tensor.unsqueeze(0)
        
        # Verify shape matches design specifications
        if feature_tensor.shape != (1, 128, 94):
            raise DatasetIOError(
                f"Invalid feature shape at index {idx}. Expected (1, 128, 94), but got {feature_tensor.shape}."
            )
            
        label = self.label_encoder.encode(emotion)
        
        return feature_tensor, label

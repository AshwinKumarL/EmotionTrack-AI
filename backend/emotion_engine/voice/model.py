from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn

from backend.emotion_engine.voice.loader import VoiceEngineError

# =====================================================================
# Exception Hierarchy
# =====================================================================

class ModelError(VoiceEngineError):
    """
    Base exception class for all errors arising within the Model module.
    Inherits from VoiceEngineError.
    """
    pass


class ModelConfigurationError(ModelError):
    """
    Exception raised when model parameters or layer configurations are invalid.
    """
    pass


# =====================================================================
# Model Configuration Data Class
# =====================================================================

@dataclass(frozen=True)
class VoiceModelConfig:
    """
    Immutable hyperparameter configuration for the EmotionCNN model.
    """
    num_classes: int
    input_channels: int
    dropout_rate: float
    filter_sizes: Tuple[int, ...]
    kernel_sizes: Tuple[int, ...]
    pool_sizes: Tuple[int, ...]
    hidden_size: int
    input_height: int = 128
    input_width: int = 94

    def __post_init__(self) -> None:
        """
        Validates configuration variables to prevent architectural issues.
        """
        # Validate types and ranges
        if not isinstance(self.num_classes, int) or self.num_classes <= 1:
            raise ModelConfigurationError("num_classes must be an integer greater than 1.")

        if not isinstance(self.input_channels, int) or self.input_channels <= 0:
            raise ModelConfigurationError("input_channels must be an integer greater than 0.")

        if not isinstance(self.dropout_rate, (int, float)) or not (0.0 <= self.dropout_rate < 1.0):
            raise ModelConfigurationError("dropout_rate must satisfy the range 0.0 <= p < 1.0.")

        if not isinstance(self.hidden_size, int) or self.hidden_size <= 0:
            raise ModelConfigurationError("hidden_size must be an integer greater than 0.")

        # Validate sequence configurations
        if not isinstance(self.filter_sizes, (list, tuple)) or not all(isinstance(f, int) and f > 0 for f in self.filter_sizes):
            raise ModelConfigurationError("filter_sizes must be a sequence of positive integers.")

        if not isinstance(self.kernel_sizes, (list, tuple)) or not all(isinstance(k, int) and k > 0 for k in self.kernel_sizes):
            raise ModelConfigurationError("kernel_sizes must be a sequence of positive integers.")

        if not isinstance(self.pool_sizes, (list, tuple)) or not all(isinstance(p, int) and p > 0 for p in self.pool_sizes):
            raise ModelConfigurationError("pool_sizes must be a sequence of positive integers.")

        # Ensure shapes align
        n_blocks = len(self.filter_sizes)
        if len(self.kernel_sizes) != n_blocks or len(self.pool_sizes) != n_blocks:
            raise ModelConfigurationError(
                f"Configuration length mismatch. filter_sizes ({n_blocks}), "
                f"kernel_sizes ({len(self.kernel_sizes)}), and pool_sizes ({len(self.pool_sizes)}) "
                "must have matching lengths."
            )


# =====================================================================
# Reusable Convolutional Block
# =====================================================================

class ConvBlock(nn.Module):
    """
    A standard Convolutional Block.
    Executes in sequence: Conv2D -> BatchNorm2D -> ReLU -> MaxPool2D.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool_size: int) -> None:
        super().__init__()
        
        # Calculate padding to preserve spatial dimensions (height/width) before pooling.
        # This assumes odd kernel sizes (which is standard practice in CNNs).
        padding = kernel_size // 2
        
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False  # Bias is redundant since batch normalization immediately follows
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


# =====================================================================
# CNN Model Architecture
# =====================================================================

class EmotionCNN(nn.Module):
    """
    Deep learning neural network for speech emotion classification from Mel Spectrograms.
    Constructs an arbitrary number of convolutional blocks dynamically from the configuration
    and dynamically flattens features using a dummy forward pass.
    """
    def __init__(self, config: VoiceModelConfig) -> None:
        super().__init__()
        
        if not isinstance(config, VoiceModelConfig):
            raise ModelConfigurationError("config must be an instance of VoiceModelConfig.")
            
        self.config = config

        # 1. Dynamically Build the Feature Extractor
        blocks = []
        in_channels = config.input_channels
        
        for i in range(len(config.filter_sizes)):
            out_channels = config.filter_sizes[i]
            kernel_size = config.kernel_sizes[i]
            pool_size = config.pool_sizes[i]
            
            blocks.append(
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    pool_size=pool_size
                )
            )
            in_channels = out_channels
            
        self.feature_extractor = nn.Sequential(*blocks)

        # 2. Determine Dynamic Flatten Size
        self.flattened_dim = self._determine_flattened_dim()

        # 3. Build the Classifier Stage
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flattened_dim, config.hidden_size),
            nn.ReLU(),
            nn.Dropout(p=config.dropout_rate),
            nn.Linear(config.hidden_size, config.num_classes)
        )

    def _determine_flattened_dim(self) -> int:
        """
        Executes a dry-run pass of a single sample through the feature extractor
        to calculate the flattened dimension dynamically on startup.
        """
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.config.input_channels, self.config.input_height, self.config.input_width)
            try:
                dummy_output = self.feature_extractor(dummy_input)
            except Exception as e:
                raise ModelConfigurationError(
                    f"Feature extractor failed during configuration validation check. "
                    f"Verify that pool_sizes or kernel_sizes are compatible. Error: {e}"
                ) from e
                
            # Check for downsampling collapse
            if dummy_output.numel() == 0 or any(dim <= 0 for dim in dummy_output.shape):
                raise ModelConfigurationError(
                    f"Feature extractor output collapsed to empty dimensions. "
                    f"Target spatial dims ({self.config.input_height}, {self.config.input_width}) were reduced below 1x1. Output shape: {dummy_output.shape}"
                )
                
            return dummy_output.numel()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Model forward pass.
        
        Args:
            x (torch.Tensor): Input batch of shape (batch_size, input_channels, input_height, input_width).
            
        Returns:
            torch.Tensor: Raw logits of shape (batch_size, num_classes).
        """
        # Validate dimensions at runtime
        if x.ndim != 4:
            raise ValueError(f"Input tensor must be 4-dimensional (B, C, H, W), got shape: {x.shape}")
            
        if x.shape[1] != self.config.input_channels:
            raise ValueError(
                f"Input channel mismatch. Expected {self.config.input_channels} channels, "
                f"but got {x.shape[1]} from tensor shape {x.shape}"
            )
            
        if x.shape[2] != self.config.input_height or x.shape[3] != self.config.input_width:
            raise ValueError(
                f"Input spatial dimension mismatch. Expected shape (B, C, {self.config.input_height}, {self.config.input_width}), "
                f"but got spatial layout ({x.shape[2]}, {x.shape[3]})"
            )

        # 1. Feature Extraction (Convolutional Blocks)
        features = self.feature_extractor(x)
        
        # 2. Classification (Flatten, Fully Connected, ReLU, Dropout, Logits)
        logits = self.classifier(features)
        
        return logits

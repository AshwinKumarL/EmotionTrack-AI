import logging
from pathlib import Path
from typing import Dict, Optional
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from backend.config import settings

logger = logging.getLogger(__name__)


class TextEmotionModel:
    """
    Manages loading and caching of the Hugging Face model and tokenizer.
    Enables choosing the appropriate computation device (CPU/GPU).
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_dir: Optional[Path] = None,
        device: Optional[str] = None,
    ):
        self.model_name = model_name or settings.TEXT_MODEL_NAME
        self.cache_dir = cache_dir or settings.MODELS_DIR
        
        # Determine device (CUDA, MPS, or CPU)
        if device:
            self.device = torch.device(device)
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
                
        self.tokenizer = None
        self.model = None
        self.id2label: Dict[int, str] = {}

    def load(self) -> None:
        """
        Loads the tokenizer and sequence classification model from Hugging Face.
        Stores them in memory and sets up model labels.
        """
        if self.tokenizer is not None and self.model is not None:
            # Model already loaded
            return

        logger.info(f"Loading tokenizer for model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=str(self.cache_dir)
        )

        logger.info(f"Loading sequence classification model: {self.model_name} on device: {self.device}")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            cache_dir=str(self.cache_dir)
        ).to(self.device)

        # Set model to evaluation mode
        self.model.eval()

        # Retrieve dynamic label mapping from model config
        if hasattr(self.model.config, "id2label") and self.model.config.id2label:
            self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        else:
            # Fallback default mapping for Ekman's model if not specified in config
            self.id2label = {
                0: "anger",
                1: "disgust",
                2: "fear",
                3: "joy",
                4: "neutral",
                5: "sadness",
                6: "surprise",
            }
        logger.info(f"Model loaded successfully with labels: {list(self.id2label.values())}")

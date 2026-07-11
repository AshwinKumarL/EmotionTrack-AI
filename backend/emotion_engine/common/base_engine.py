from abc import ABC, abstractmethod
from typing import Any


class BaseEmotionEngine(ABC):
    """
    Abstract base class for all emotion processing engines.
    Provides a standard interface for model loading and emotion inference.
    """

    @abstractmethod
    def load_model(self) -> None:
        """
        Load and initialize the model assets (weights, tokenizers, files)
        necessary for executing inference.
        """
        pass

    @abstractmethod
    def predict(self, inputs: Any) -> Any:
        """
        Execute prediction pipeline: preprocess input, run inference,
        and postprocess output into standard formats.
        
        Args:
            inputs: The modality input data (e.g. string text, audio array, etc.).
            
        Returns:
            The standardized prediction result (e.g., Pydantic schema).
        """
        pass

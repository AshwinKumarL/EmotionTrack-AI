import logging
from typing import Optional
import torch

from backend.emotion_engine.common.base_engine import BaseEmotionEngine
from backend.emotion_engine.text.preprocess import TextPreprocessor
from backend.emotion_engine.text.model import TextEmotionModel
from backend.emotion_engine.text.schemas import TextEmotionResponse

logger = logging.getLogger(__name__)


class TextEmotionEngine(BaseEmotionEngine):
    """
    Inference engine that coordinates preprocessing, tokenization, model forward pass,
    and post-processing for text emotion understanding.
    """

    def __init__(
        self,
        preprocessor: Optional[TextPreprocessor] = None,
        model_wrapper: Optional[TextEmotionModel] = None,
    ):
        self.preprocessor = preprocessor or TextPreprocessor()
        self.model_wrapper = model_wrapper or TextEmotionModel()

    def load_model(self) -> None:
        """
        Loads the underlying transformer model and tokenizer.
        """
        self.model_wrapper.load()

    def predict(self, text: str) -> TextEmotionResponse:
        """
        Analyzes the emotional state from the input text.
        
        Args:
            text: Raw input text.
            
        Returns:
            TextEmotionResponse containing primary emotion, confidence, and probabilities.
        """
        # 1. Preprocess
        cleaned_text = self.preprocessor.clean(text)
        if not cleaned_text:
            raise ValueError("Input text is empty after preprocessing.")

        # 2. Lazy load model if not preloaded
        if self.model_wrapper.tokenizer is None or self.model_wrapper.model is None:
            logger.info("Model not pre-loaded; loading dynamically on first prediction request.")
            self.load_model()

        # 3. Tokenize input
        inputs = self.model_wrapper.tokenizer(
            cleaned_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,  # Set a sensible max length for DistilRoBERTa
        ).to(self.model_wrapper.device)

        # 4. Perform Inference
        with torch.no_grad():
            outputs = self.model_wrapper.model(**inputs)
            
        # 5. Extract probabilities using Softmax
        logits = outputs.logits
        probabilities_tensor = torch.softmax(logits, dim=-1).squeeze()
        
        # Handle case where probabilities is a list (normal) vs single element (unlikely)
        if probabilities_tensor.ndim == 0:
            probs = [probabilities_tensor.item()]
        else:
            probs = probabilities_tensor.tolist()

        # 6. Map predictions to label dictionary
        probabilities_dict = {}
        max_prob = -1.0
        primary_label = "neutral"

        for idx, prob in enumerate(probs):
            label_name = self.model_wrapper.id2label.get(idx, f"label_{idx}").lower()
            # Standardize: round to 4 decimal places for clean APIs
            rounded_prob = round(prob, 4)
            probabilities_dict[label_name] = rounded_prob
            
            if rounded_prob > max_prob:
                max_prob = rounded_prob
                primary_label = label_name

        # 7. Construct and return response
        # Standardize primary emotion as capitalized (e.g. "sadness" -> "Sadness")
        return TextEmotionResponse(
            primary_emotion=primary_label.capitalize(),
            confidence=max_prob,
            probabilities=probabilities_dict
        )

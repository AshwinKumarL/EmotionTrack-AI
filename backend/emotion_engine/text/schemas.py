from typing import Dict
from pydantic import BaseModel, Field, field_validator


class TextEmotionRequest(BaseModel):
    """
    Request body schema for text emotion analysis.
    """
    text: str = Field(
        ...,
        description="The raw text input to analyze for emotions.",
        examples=["I don't feel like talking today."]
    )

    @field_validator("text")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        """
        Verify that the input text is not empty or just whitespace.
        """
        if not value or not value.strip():
            raise ValueError("Input text cannot be empty or only whitespace.")
        return value.strip()


class TextEmotionResponse(BaseModel):
    """
    Response schema representing the emotion understanding output.
    """
    primary_emotion: str = Field(
        ...,
        description="The emotion category with the highest probability.",
        examples=["Sadness"]
    )
    confidence: float = Field(
        ...,
        description="The confidence score of the primary emotion prediction.",
        examples=[0.92]
    )
    probabilities: Dict[str, float] = Field(
        ...,
        description="The probability scores distribution across all analyzed emotions.",
        examples=[{
            "joy": 0.02,
            "sadness": 0.92,
            "anger": 0.01,
            "fear": 0.03,
            "neutral": 0.02
        }]
    )

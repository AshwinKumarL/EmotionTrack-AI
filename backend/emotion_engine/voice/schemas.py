from typing import Dict, Optional
from pydantic import BaseModel, Field, field_validator


class VoiceEmotionRequest(BaseModel):
    """
    Request body schema for voice emotion analysis.
    Accepts base64-encoded audio data.
    """
    audio_base64: str = Field(
        ...,
        description="Base64-encoded audio data (WAV, FLAC, or MP3 format).",
        examples=["UklGRiQAAABXQVZFZm10IBAAAA..."]
    )
    filename: Optional[str] = Field(
        default=None,
        description="Original filename with extension (used for format detection).",
        examples=["sample.wav"]
    )

    @field_validator("audio_base64")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Audio data cannot be empty.")
        return value.strip()


class VoiceEmotionResponse(BaseModel):
    """
    Response schema representing the voice emotion analysis output.
    """
    primary_emotion: str = Field(
        ...,
        description="The emotion category with the highest probability.",
        examples=["Happy"]
    )
    confidence: float = Field(
        ...,
        description="The confidence score of the primary emotion prediction.",
        examples=[0.87]
    )
    probabilities: Dict[str, float] = Field(
        ...,
        description="Probability distribution across all emotion classes.",
        examples=[{
            "happy": 0.87,
            "sad": 0.03,
            "angry": 0.02,
            "fearful": 0.05,
            "calm": 0.03
        }]
    )
    model_version: str = Field(
        default="v2",
        description="The model version used for prediction.",
        examples=["v2"]
    )

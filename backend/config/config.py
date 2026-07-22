import os
from pathlib import Path
from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base Directory: root of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """
    Application settings and configurations.
    Values can be overridden using environment variables or a .env file.
    """
    # API Configurations
    PROJECT_NAME: str = "EmotionAI Multi-Modal Engine"
    API_V1_STR: str = "/api/v1"
    ALLOWED_ORIGINS: List[str] = ["*"]

    # Model Configurations
    TEXT_MODEL_NAME: str = "j-hartmann/emotion-english-distilroberta-base"
    VOICE_MODEL_VERSION: str = "v2"
    VOICE_MODEL_FILENAME: str = "best_voice_model_v2.pth"

    # Directory to store cached models offline
    MODELS_DIR: Path = BASE_DIR / "models"

    # Confidence threshold to filter out low confidence emotion predictions
    CONFIDENCE_THRESHOLD: float = 0.0

    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


# Instantiate settings for global use
settings = Settings()

# Ensure the models directory exists
os.makedirs(settings.MODELS_DIR, exist_ok=True)

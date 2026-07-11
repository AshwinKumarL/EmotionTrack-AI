import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

# Base Directory: root of the project
BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """
    Application settings and configurations.
    Values can be overridden using environment variables or a .env file.
    """
    # API Configurations
    PROJECT_NAME: str = "EmotionAI Text Engine"
    API_V1_STR: str = "/api/v1"
    
    # Model Configurations
    # Default model fine-tuned on Ekman's basic emotions plus neutral
    TEXT_MODEL_NAME: str = "j-hartmann/emotion-english-distilroberta-base"
    
    # Directory to store cached models offline
    MODELS_DIR: Path = BASE_DIR / "models"
    
    # Confidence threshold to filter out low confidence emotion predictions
    CONFIDENCE_THRESHOLD: float = 0.0

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


# Instantiate settings for global use
settings = Settings()

# Ensure the models directory exists
os.makedirs(settings.MODELS_DIR, exist_ok=True)

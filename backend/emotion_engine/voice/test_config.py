import os
from pathlib import Path

from backend.config.config import Settings, BASE_DIR


def test_base_dir_resolves_to_project_root():
    assert BASE_DIR.name == "aswin"
    assert BASE_DIR.exists()


def test_default_project_name():
    s = Settings()
    assert s.PROJECT_NAME == "EmotionAI Multi-Modal Engine"


def test_default_api_prefix():
    s = Settings()
    assert s.API_V1_STR == "/api/v1"


def test_default_allowed_origins():
    s = Settings()
    assert s.ALLOWED_ORIGINS == ["*"]


def test_models_dir_is_path():
    s = Settings()
    assert isinstance(s.MODELS_DIR, Path)


def test_voice_model_defaults():
    s = Settings()
    assert s.VOICE_MODEL_VERSION == "v2"
    assert s.VOICE_MODEL_FILENAME == "best_voice_model_v2.pth"

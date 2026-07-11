import pytest
from pydantic import ValidationError

from backend.emotion_engine.text.preprocess import TextPreprocessor
from backend.emotion_engine.text.schemas import TextEmotionRequest, TextEmotionResponse
from backend.emotion_engine.text.inference import TextEmotionEngine


def test_preprocessor_strips_whitespace():
    preprocessor = TextPreprocessor()
    assert preprocessor.clean("  hello world  ") == "hello world"


def test_preprocessor_removes_extra_spaces():
    preprocessor = TextPreprocessor(remove_extra_spaces=True)
    assert preprocessor.clean("hello    world\n\nnext  line") == "hello world next line"


def test_preprocessor_handles_empty_input():
    preprocessor = TextPreprocessor()
    assert preprocessor.clean("") == ""


def test_schema_validates_non_empty_text():
    # Valid text
    req = TextEmotionRequest(text="Hello world")
    assert req.text == "Hello world"

    # Empty text raises validation error
    with pytest.raises(ValidationError):
        TextEmotionRequest(text="")

    # Whitespace-only text raises validation error
    with pytest.raises(ValidationError):
        TextEmotionRequest(text="    ")


def test_schema_serialization():
    data = {
        "primary_emotion": "Sadness",
        "confidence": 0.92,
        "probabilities": {
            "joy": 0.02,
            "sadness": 0.92,
            "anger": 0.01,
            "fear": 0.03,
            "neutral": 0.02
        }
    }
    response = TextEmotionResponse(**data)
    assert response.primary_emotion == "Sadness"
    assert response.confidence == 0.92
    assert response.probabilities["sadness"] == 0.92


@pytest.mark.slow
def test_engine_inference():
    """
    Integration test checking actual inference.
    Will download the model on first run and cache it.
    """
    engine = TextEmotionEngine()
    
    # Pre-load to isolate load time from test measurement
    engine.load_model()
    
    input_text = "I don't feel like talking today."
    response = engine.predict(input_text)
    
    # Assert return types and schemas
    assert isinstance(response, TextEmotionResponse)
    assert response.primary_emotion in [e.capitalize() for e in engine.model_wrapper.id2label.values()]
    assert 0.0 <= response.confidence <= 1.0
    assert len(response.probabilities) == len(engine.model_wrapper.id2label)
    
    # Let's verify sadness is the primary emotion for "I don't feel like talking today."
    # using j-hartmann model
    assert response.primary_emotion == "Sadness"

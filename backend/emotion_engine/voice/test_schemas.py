import pytest
from pydantic import ValidationError

from backend.emotion_engine.voice.schemas import VoiceEmotionRequest, VoiceEmotionResponse


class TestVoiceEmotionRequest:
    def test_valid_request(self):
        req = VoiceEmotionRequest(audio_base64="UklGRiQAAABXQVZFZm10IBAAAA")
        assert req.audio_base64 == "UklGRiQAAABXQVZFZm10IBAAAA"
        assert req.filename is None

    def test_valid_request_with_filename(self):
        req = VoiceEmotionRequest(audio_base64="dGVzdA==", filename="sample.wav")
        assert req.filename == "sample.wav"

    def test_empty_string_raises(self):
        with pytest.raises(ValidationError, match="cannot be empty"):
            VoiceEmotionRequest(audio_base64="")

    def test_whitespace_only_raises(self):
        with pytest.raises(ValidationError, match="cannot be empty"):
            VoiceEmotionRequest(audio_base64="   ")

    def test_strips_whitespace(self):
        req = VoiceEmotionRequest(audio_base64="  dGVzdA==  ")
        assert req.audio_base64 == "dGVzdA=="

    def test_missing_field_raises(self):
        with pytest.raises(ValidationError):
            VoiceEmotionRequest()


class TestVoiceEmotionResponse:
    def test_construction(self):
        resp = VoiceEmotionResponse(
            primary_emotion="Happy",
            confidence=0.87,
            probabilities={"happy": 0.87, "sad": 0.03, "angry": 0.02, "fearful": 0.05, "calm": 0.03},
        )
        assert resp.primary_emotion == "Happy"
        assert resp.confidence == 0.87
        assert resp.model_version == "v2"

    def test_model_version_default(self):
        resp = VoiceEmotionResponse(
            primary_emotion="Calm",
            confidence=0.95,
            probabilities={"calm": 0.95},
        )
        assert resp.model_version == "v2"

    def test_serialization_roundtrip(self):
        resp = VoiceEmotionResponse(
            primary_emotion="Sad",
            confidence=0.72,
            probabilities={"sad": 0.72, "happy": 0.10},
        )
        d = resp.model_dump()
        assert d["primary_emotion"] == "Sad"
        assert d["confidence"] == 0.72
        assert "sad" in d["probabilities"]

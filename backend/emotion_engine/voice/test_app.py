import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from fastapi.testclient import TestClient

from backend.emotion_engine.voice.schemas import VoiceEmotionResponse


@pytest.fixture
def mock_engines():
    with patch("backend.app.text_engine") as text_eng, \
         patch("backend.app.voice_engine") as voice_eng:
        text_eng.predict = MagicMock(return_value=MagicMock(
            primary_emotion="Sadness",
            confidence=0.92,
            probabilities={"sadness": 0.92, "joy": 0.03, "anger": 0.01, "fear": 0.02, "neutral": 0.02}
        ))
        voice_eng.predict = MagicMock(return_value=MagicMock(
            primary_emotion="Happy",
            confidence=0.87,
            probabilities={"happy": 0.87, "sad": 0.03, "angry": 0.02, "fearful": 0.05, "calm": 0.03},
            model_version="v2"
        ))
        voice_eng.decode_base64_audio = MagicMock(return_value=b"fake-audio-bytes")
        yield text_eng, voice_eng


@pytest.fixture
def client(mock_engines):
    from backend.app import app
    return TestClient(app, raise_server_exceptions=False)


class TestHealthCheck:
    def test_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "text_model" in data
        assert "voice_model" in data


class TestAnalyzeText:
    def test_success(self, client, mock_engines):
        text_eng, _ = mock_engines
        resp = client.post("/api/v1/text/analyze", json={"text": "I feel sad today"})
        assert resp.status_code == 200
        data = resp.json()
        assert "primary_emotion" in data
        assert "confidence" in data
        assert "probabilities" in data
        text_eng.predict.assert_called_once_with("I feel sad today")

    def test_empty_text_returns_422(self, client):
        resp = client.post("/api/v1/text/analyze", json={"text": ""})
        assert resp.status_code == 422

    def test_missing_field_returns_422(self, client):
        resp = client.post("/api/v1/text/analyze", json={})
        assert resp.status_code == 422

    def test_engine_error_returns_500(self, client, mock_engines):
        text_eng, _ = mock_engines
        text_eng.predict.side_effect = RuntimeError("model broken")
        resp = client.post("/api/v1/text/analyze", json={"text": "hello"})
        assert resp.status_code == 500
        assert "Internal server error" in resp.json()["detail"]


class TestAnalyzeVoice:
    def test_success(self, client, mock_engines):
        _, voice_eng = mock_engines
        resp = client.post(
            "/api/v1/voice/analyze",
            files={"file": ("test.wav", b"fake-wav-data", "audio/wav")}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "primary_emotion" in data
        assert data["model_version"] == "v2"

    def test_empty_file_returns_400(self, client):
        resp = client.post(
            "/api/v1/voice/analyze",
            files={"file": ("empty.wav", b"", "audio/wav")}
        )
        assert resp.status_code == 400

    def test_engine_error_returns_500(self, client, mock_engines):
        _, voice_eng = mock_engines
        voice_eng.predict.side_effect = RuntimeError("inference failed")
        resp = client.post(
            "/api/v1/voice/analyze",
            files={"file": ("test.wav", b"fake-wav-data", "audio/wav")}
        )
        assert resp.status_code == 500
        assert "Internal server error" in resp.json()["detail"]


class TestAnalyzeVoiceBase64:
    def test_success(self, client, mock_engines):
        _, voice_eng = mock_engines
        resp = client.post(
            "/api/v1/voice/analyze-base64",
            json={"audio_base64": "dGVzdA=="}
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "primary_emotion" in data

    def test_empty_base64_returns_422(self, client):
        resp = client.post(
            "/api/v1/voice/analyze-base64",
            json={"audio_base64": ""}
        )
        assert resp.status_code == 422

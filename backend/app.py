import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.config.config import settings
from backend.emotion_engine.text.inference import TextEmotionEngine
from backend.emotion_engine.text.schemas import TextEmotionRequest, TextEmotionResponse
from backend.emotion_engine.voice.inference import VoiceEmotionEngine
from backend.emotion_engine.voice.schemas import VoiceEmotionRequest, VoiceEmotionResponse

logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parent.parent
VM2_MODEL_PATH = ROOT_DIR / "models" / "best_voice_model_v2.pth"

text_engine: TextEmotionEngine = None
voice_engine: VoiceEmotionEngine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global text_engine, voice_engine

    logger.info("Loading text emotion engine...")
    text_engine = TextEmotionEngine()
    text_engine.load_model()

    logger.info("Loading voice emotion engine (VM/2)...")
    voice_engine = VoiceEmotionEngine(model_path=VM2_MODEL_PATH)
    voice_engine.load_model()

    logger.info("Both engines loaded. API ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Multi-modal Emotion Recognition API — Text and Voice",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "text_model": settings.TEXT_MODEL_NAME,
        "voice_model": "VM/2 (best_voice_model_v2.pth)",
    }


@app.post(f"{settings.API_V1_STR}/text/analyze", response_model=TextEmotionResponse)
def analyze_text(request: TextEmotionRequest):
    try:
        return text_engine.predict(request.text)
    except Exception as e:
        logger.error(f"Text analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during text analysis.")


@app.post(f"{settings.API_V1_STR}/voice/analyze", response_model=VoiceEmotionResponse)
async def analyze_voice(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file uploaded.")
        return voice_engine.predict(audio_bytes)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during voice analysis.")


@app.post(f"{settings.API_V1_STR}/voice/analyze-base64", response_model=VoiceEmotionResponse)
def analyze_voice_base64(request: VoiceEmotionRequest):
    try:
        audio_bytes = VoiceEmotionEngine.decode_base64_audio(request.audio_base64)
        return voice_engine.predict(audio_bytes)
    except Exception as e:
        logger.error(f"Voice base64 analysis failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during voice analysis.")

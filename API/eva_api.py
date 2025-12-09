"""
EVA API Server - FastAPI Backend

Exposes the EVA pipeline as a REST API for the React Native app.

Endpoints:
    POST /process          - Process audio file, return full result
    POST /process/stream   - Process with streaming LLM response
    GET  /health           - Health check
    GET  /config           - Get current configuration
"""

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Dict, List
from contextlib import asynccontextmanager
import asyncio

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------
# Response Models
# --------------------------
class EmotionScore(BaseModel):
    """Single emotion score"""
    emotion: str
    score: float
    percentage: str


class ProcessingTimes(BaseModel):
    """Processing time breakdown"""
    stt: float
    ser: float
    llm: Optional[float] = None
    total: float


class ProcessResponse(BaseModel):
    """Full processing response"""
    success: bool

    # Core results
    transcription: str
    language: str
    stt_confidence: Optional[float]

    # Emotions
    emotions: List[EmotionScore]
    primary_emotion: str
    primary_emotion_score: float

    # LLM response
    eva_response: Optional[str] = None
    llm_model: Optional[str] = None

    # Metadata
    processing_times: ProcessingTimes
    audio_duration: Optional[float] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    components: Dict[str, bool]


class ConfigResponse(BaseModel):
    """Configuration response"""
    stt_model: str
    stt_backend: str
    language: str
    ser_model_loaded: bool
    llm_backend: Optional[str]
    llm_model: Optional[str]
    llm_available: bool


class ErrorResponse(BaseModel):
    """Error response"""
    success: bool = False
    error: str
    detail: Optional[str] = None


# --------------------------
# Global Pipeline Instance
# --------------------------
pipeline = None


def init_pipeline():
    """Initialize the EVA pipeline"""
    global pipeline

    # Import here to avoid circular imports
    try:
        from Pipeline.eva_pipeline import EVAPipeline
    except ImportError:
        from eva_pipeline import EVAPipeline

    checkpoint_path = os.getenv(
        "SER_CHECKPOINT",
        str(PROJECT_ROOT / "checkpoints" / "best_model.pth")
    )

    stt_model = os.getenv("STT_MODEL", "base")
    language = os.getenv("LANGUAGE", "vi")
    llm_backend = os.getenv("LLM_BACKEND", None)
    llm_model = os.getenv("LLM_MODEL", None)

    print(f"\n🚀 Initializing EVA API Server")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   STT Model: {stt_model}")
    print(f"   Language: {language}")

    pipeline = EVAPipeline(
        ser_checkpoint=checkpoint_path,
        stt_model=stt_model,
        language=language,
        parallel=True,
        enable_llm=True,
        llm_backend=llm_backend,
        llm_model=llm_model
    )

    return pipeline


# --------------------------
# Lifespan Manager
# --------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    print("\n" + "=" * 60)
    print("🎭 EVA API Server Starting...")
    print("=" * 60)

    init_pipeline()

    print("\n✅ Server ready!")
    print("=" * 60 + "\n")

    yield

    # Shutdown
    print("\n👋 EVA API Server shutting down...")


# --------------------------
# FastAPI App
# --------------------------
app = FastAPI(
    title="EVA API",
    description="Empathic Voice Assistant - Backend API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware for React Native
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your app's origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------
# Helper Functions
# --------------------------
def format_emotions(emotions: Dict[str, float]) -> List[EmotionScore]:
    """Format emotions dict into sorted list"""
    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
    return [
        EmotionScore(
            emotion=emotion,
            score=round(score, 4),
            percentage=f"{score * 100:.1f}%"
        )
        for emotion, score in sorted_emotions
    ]


async def process_audio_file(file: UploadFile, generate_response: bool = True) -> ProcessResponse:
    """Process uploaded audio file"""
    global pipeline

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    # Save uploaded file temporarily
    suffix = Path(file.filename).suffix if file.filename else ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Process through pipeline
        result = pipeline.process(tmp_path, generate_response=generate_response)

        # Format emotions
        emotions_list = format_emotions(result.emotions)
        primary = emotions_list[0] if emotions_list else None

        # Build response
        response = ProcessResponse(
            success=True,
            transcription=result.text,
            language=result.stt_result.language,
            stt_confidence=result.stt_confidence,
            emotions=emotions_list,
            primary_emotion=primary.emotion if primary else "Unknown",
            primary_emotion_score=primary.score if primary else 0.0,
            eva_response=result.llm_response,
            llm_model=result.llm_result.model if result.llm_result else None,
            processing_times=ProcessingTimes(
                stt=round(result.stt_result.processing_time, 3),
                ser=round(result.ser_result.processing_time, 3),
                llm=round(result.llm_result.latency, 3) if result.llm_result else None,
                total=round(result.total_processing_time, 3)
            ),
            audio_duration=len(content) / 32000  # Rough estimate
        )

        return response

    finally:
        # Cleanup temp file
        os.unlink(tmp_path)


# --------------------------
# API Endpoints
# --------------------------
@app.get("/", response_model=Dict)
async def root():
    """Root endpoint"""
    return {
        "name": "EVA API",
        "version": "1.0.0",
        "description": "Empathic Voice Assistant Backend",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    global pipeline

    components = {
        "pipeline": pipeline is not None,
        "stt": pipeline is not None and pipeline.stt is not None,
        "ser": pipeline is not None and pipeline.ser is not None,
        "llm": pipeline is not None and pipeline.llm is not None and pipeline.llm.is_available()
    }

    status = "healthy" if all([components["pipeline"], components["stt"], components["ser"]]) else "degraded"

    return HealthResponse(
        status=status,
        version="1.0.0",
        components=components
    )


@app.get("/config", response_model=ConfigResponse)
async def get_config():
    """Get current configuration"""
    global pipeline

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    return ConfigResponse(
        stt_model=pipeline.stt.engine.model_size if hasattr(pipeline.stt.engine, 'model_size') else "unknown",
        stt_backend="whisper",
        language=pipeline.language,
        ser_model_loaded=pipeline.ser is not None,
        llm_backend=pipeline.llm.backend_name if pipeline.llm else None,
        llm_model=pipeline.llm._backend.model if pipeline.llm and pipeline.llm._backend else None,
        llm_available=pipeline.llm is not None and pipeline.llm.is_available()
    )


@app.post("/process", response_model=ProcessResponse)
async def process_audio(
        file: UploadFile = File(..., description="Audio file to process"),
        generate_response: bool = True
):
    """
    Process audio file through EVA pipeline

    - **file**: Audio file (WAV, MP3, etc.)
    - **generate_response**: Whether to generate LLM response (default: True)

    Returns transcription, emotions, and EVA's empathic response.
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Check file extension
    allowed_extensions = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}
    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {allowed_extensions}"
        )

    try:
        return await process_audio_file(file, generate_response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe", response_model=Dict)
async def transcribe_only(file: UploadFile = File(...)):
    """
    Transcribe audio without emotion analysis or LLM response

    Faster endpoint for just getting the text.
    """
    global pipeline

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    # Save temp file
    suffix = Path(file.filename).suffix if file.filename else ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Just run STT
        result = pipeline.stt.transcribe_file(tmp_path)

        return {
            "success": True,
            "transcription": result.text,
            "language": result.language,
            "confidence": result.confidence,
            "processing_time": round(result.processing_time, 3)
        }
    finally:
        os.unlink(tmp_path)


@app.post("/emotions", response_model=Dict)
async def emotions_only(file: UploadFile = File(...)):
    """
    Analyze emotions without transcription or LLM response

    Faster endpoint for just getting emotion scores.
    """
    global pipeline

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    import librosa

    # Save temp file
    suffix = Path(file.filename).suffix if file.filename else ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Load audio and run SER
        audio, sr = librosa.load(tmp_path, sr=16000, mono=True)
        result = pipeline.ser.predict(audio, sr)

        emotions_list = format_emotions(result.emotions)
        primary = emotions_list[0] if emotions_list else None

        return {
            "success": True,
            "emotions": [e.dict() for e in emotions_list],
            "primary_emotion": primary.emotion if primary else "Unknown",
            "primary_emotion_score": primary.score if primary else 0.0,
            "processing_time": round(result.processing_time, 3)
        }
    finally:
        os.unlink(tmp_path)


@app.post("/chat", response_model=Dict)
async def chat_text(
        text: str,
        emotion: Optional[str] = None,
        emotion_score: Optional[float] = None
):
    """
    Chat with EVA using text input (no audio)

    Optionally provide emotion context manually.
    """
    global pipeline

    if pipeline is None or pipeline.llm is None:
        raise HTTPException(status_code=503, detail="LLM not available")

    # Build simple prompt
    if emotion and emotion_score:
        emotion_context = f"User's emotional state: {emotion} ({emotion_score * 100:.0f}%)"
    else:
        emotion_context = "User's emotional state: Unknown"

    prompt = f"""{pipeline.prompt_manager.system_context}

{emotion_context}

User's message: {text}

Your empathic response:"""

    try:
        start_time = time.time()
        response = pipeline.llm.generate(prompt, max_tokens=256, temperature=0.7)
        latency = time.time() - start_time

        return {
            "success": True,
            "response": response.text,
            "model": response.model,
            "processing_time": round(latency, 3)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload-prompts")
async def reload_prompts():
    """Reload prompt templates from files (hot-reload)"""
    global pipeline

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    try:
        pipeline.prompt_manager.reload()
        return {"success": True, "message": "Prompts reloaded"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------
# Error Handlers
# --------------------------
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            success=False,
            error=exc.detail
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            success=False,
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )


# --------------------------
# Main Entry Point
# --------------------------
def main():
    """Run the server"""
    import argparse

    parser = argparse.ArgumentParser(description="EVA API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")

    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    EVA API Server                            ║
╠══════════════════════════════════════════════════════════════╣
║  Endpoints:                                                  ║
║    POST /process     - Full pipeline (STT + SER + LLM)       ║
║    POST /transcribe  - Speech-to-text only                   ║
║    POST /emotions    - Emotion analysis only                 ║
║    POST /chat        - Text chat with EVA                    ║
║    GET  /health      - Health check                          ║
║    GET  /config      - Current configuration                 ║
║    GET  /docs        - Interactive API docs                  ║
╚══════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "eva_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1
    )


if __name__ == "__main__":
    main()
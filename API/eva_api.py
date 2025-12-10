"""
EVA API Server - FastAPI Backend with Conversation History

Exposes the EVA pipeline as a REST API for the React Native app.
Now supports conversation context for multi-turn dialogues.

Endpoints:
    POST /process          - Process audio file, return full result
    POST /chat             - Text chat with EVA
    POST /conversation/new - Start a new conversation
    DELETE /conversation   - Clear conversation history
    GET  /conversation     - Get current conversation history
    GET  /health           - Health check
    GET  /config           - Get current configuration
"""

import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional, Dict, List
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------
# Conversation Storage
# --------------------------
class ConversationMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    emotion: Optional[str] = None
    emotion_score: Optional[float] = None
    timestamp: str


class ConversationStore:
    """
    Simple in-memory conversation storage.
    In production, use Redis or a database.
    """

    def __init__(self, max_history: int = 20):
        self.conversations: Dict[str, List[ConversationMessage]] = {}
        self.max_history = max_history

    def get_or_create(self, session_id: str) -> List[ConversationMessage]:
        if session_id not in self.conversations:
            self.conversations[session_id] = []
        return self.conversations[session_id]

    def add_message(self, session_id: str, message: ConversationMessage):
        history = self.get_or_create(session_id)
        history.append(message)

        # Trim to max history
        if len(history) > self.max_history:
            self.conversations[session_id] = history[-self.max_history:]

    def clear(self, session_id: str):
        if session_id in self.conversations:
            self.conversations[session_id] = []

    def delete(self, session_id: str):
        if session_id in self.conversations:
            del self.conversations[session_id]

    def get_history(self, session_id: str) -> List[ConversationMessage]:
        return self.get_or_create(session_id)


# Global conversation store
conversation_store = ConversationStore(max_history=20)


# --------------------------
# Response Models
# --------------------------
class EmotionScore(BaseModel):
    emotion: str
    score: float
    percentage: str


class ProcessingTimes(BaseModel):
    stt: float
    ser: float
    llm: Optional[float] = None
    total: float


class ProcessResponse(BaseModel):
    success: bool
    transcription: str
    language: str
    stt_confidence: Optional[float]
    emotions: List[EmotionScore]
    primary_emotion: str
    primary_emotion_score: float
    eva_response: Optional[str] = None
    llm_model: Optional[str] = None
    processing_times: ProcessingTimes
    audio_duration: Optional[float] = None
    session_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    components: Dict[str, bool]


class ConfigResponse(BaseModel):
    stt_model: str
    stt_backend: str
    language: str
    ser_model_loaded: bool
    llm_backend: Optional[str]
    llm_model: Optional[str]
    llm_available: bool


class ConversationResponse(BaseModel):
    session_id: str
    messages: List[ConversationMessage]
    message_count: int


class ErrorResponse(BaseModel):
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
    print("\n" + "=" * 60)
    print("🎭 EVA API Server Starting...")
    print("=" * 60)

    init_pipeline()

    print("\n✅ Server ready!")
    print("=" * 60 + "\n")

    yield

    print("\n👋 EVA API Server shutting down...")


# --------------------------
# FastAPI App
# --------------------------
app = FastAPI(
    title="EVA API",
    description="Empathic Voice Assistant - Backend API with Conversation History",
    version="1.1.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------
# Helper Functions
# --------------------------
def format_emotions(emotions: Dict[str, float]) -> List[EmotionScore]:
    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
    return [
        EmotionScore(
            emotion=emotion,
            score=round(score, 4),
            percentage=f"{score * 100:.1f}%"
        )
        for emotion, score in sorted_emotions
    ]


def get_session_id(x_session_id: Optional[str] = None) -> str:
    """Get or generate session ID"""
    if x_session_id:
        return x_session_id
    return str(uuid.uuid4())


def format_conversation_for_llm(history: List[ConversationMessage], max_messages: int = 10) -> str:
    """Format conversation history for LLM context"""
    if not history:
        return ""

    # Take last N messages
    recent = history[-max_messages:]

    formatted = []
    for msg in recent:
        role_label = "User" if msg.role == "user" else "EVA"
        emotion_note = ""
        if msg.emotion and msg.role == "user":
            emotion_note = f" [feeling {msg.emotion}]"
        formatted.append(f"{role_label}{emotion_note}: {msg.content}")

    return "\n".join(formatted)


def generate_prompt_with_history(
    user_text: str,
    emotions: Dict[str, float],
    history: List[ConversationMessage],
    system_context: str,
    guidelines: str
) -> str:
    """Generate LLM prompt including conversation history"""

    # Get top emotions
    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
    top_emotions = sorted_emotions[:3]

    emotion_desc = ", ".join([
        f"{emotion} ({prob * 100:.0f}%)"
        for emotion, prob in top_emotions
    ])

    primary_emotion = top_emotions[0][0] if top_emotions else "Neutral"

    # Format conversation history
    history_text = format_conversation_for_llm(history)

    history_section = ""
    if history_text:
        history_section = f"""
═══════════════════════════════════════════════════════════════════════
CONVERSATION HISTORY
═══════════════════════════════════════════════════════════════════════
{history_text}
"""

    prompt = f"""{system_context}
{history_section}
═══════════════════════════════════════════════════════════════════════
CURRENT EMOTIONAL STATE
═══════════════════════════════════════════════════════════════════════
Detected emotions: {emotion_desc}
Primary emotion: {primary_emotion}

═══════════════════════════════════════════════════════════════════════
RESPONSE GUIDELINES
═══════════════════════════════════════════════════════════════════════
{guidelines}
- Consider the conversation history when responding
- Reference previous topics naturally if relevant
- Maintain continuity in the conversation

═══════════════════════════════════════════════════════════════════════
USER'S CURRENT MESSAGE
═══════════════════════════════════════════════════════════════════════
{user_text}

═══════════════════════════════════════════════════════════════════════
YOUR EMPATHIC RESPONSE:
"""

    return prompt


# --------------------------
# API Endpoints
# --------------------------
@app.get("/", response_model=Dict)
async def root():
    return {
        "name": "EVA API",
        "version": "1.1.0",
        "description": "Empathic Voice Assistant Backend with Conversation History",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
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
        version="1.1.0",
        components=components
    )


@app.get("/config", response_model=ConfigResponse)
async def get_config():
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
    generate_response: bool = True,
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID")
):
    """
    Process audio file through EVA pipeline with conversation history.

    Headers:
        X-Session-ID: Optional session ID for conversation continuity
    """
    global pipeline

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    allowed_extensions = {".wav", ".mp3", ".m4a", ".ogg", ".flac", ".webm"}
    suffix = Path(file.filename).suffix.lower()

    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {allowed_extensions}"
        )

    # Get or create session
    session_id = get_session_id(x_session_id)
    history = conversation_store.get_history(session_id)

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        start_time = time.time()

        # Load audio for processing
        import librosa
        audio, sr = librosa.load(tmp_path, sr=16000, mono=True)

        # STT - use the numpy array directly
        stt_start = time.time()
        stt_result = pipeline.stt.transcribe(audio, sr)
        stt_time = time.time() - stt_start

        # SER
        ser_start = time.time()
        ser_result = pipeline.ser.predict(audio, sr)
        ser_time = time.time() - ser_start

        # Format emotions
        emotions_list = format_emotions(ser_result.emotions)
        primary = emotions_list[0] if emotions_list else None
        primary_emotion = primary.emotion if primary else "Unknown"
        primary_score = primary.score if primary else 0.0

        # Add user message to history
        user_message = ConversationMessage(
            role="user",
            content=stt_result.text,
            emotion=primary_emotion,
            emotion_score=primary_score,
            timestamp=datetime.now().isoformat()
        )
        conversation_store.add_message(session_id, user_message)

        # Generate LLM response with history
        eva_response = None
        llm_model = None
        llm_time = None

        if generate_response and pipeline.llm and pipeline.llm.is_available():
            llm_start = time.time()

            # Get guidelines based on emotion
            guidelines = pipeline.prompt_manager.get_guidelines_for_emotion(
                primary_emotion,
                primary_score
            )

            # Generate prompt with conversation history
            prompt = generate_prompt_with_history(
                user_text=stt_result.text,
                emotions=ser_result.emotions,
                history=history,  # History before current message
                system_context=pipeline.prompt_manager.system_context,
                guidelines=guidelines + "\n" + pipeline.prompt_manager.general_principles
            )

            try:
                response = pipeline.llm.generate(prompt, max_tokens=256, temperature=0.7)
                eva_response = response.text
                llm_model = response.model

                # Add EVA response to history
                eva_message = ConversationMessage(
                    role="assistant",
                    content=eva_response,
                    timestamp=datetime.now().isoformat()
                )
                conversation_store.add_message(session_id, eva_message)

            except Exception as e:
                print(f"LLM error: {e}")

            llm_time = time.time() - llm_start

        total_time = time.time() - start_time

        return ProcessResponse(
            success=True,
            transcription=stt_result.text,
            language=stt_result.language,
            stt_confidence=stt_result.confidence,
            emotions=emotions_list,
            primary_emotion=primary_emotion,
            primary_emotion_score=primary_score,
            eva_response=eva_response,
            llm_model=llm_model,
            processing_times=ProcessingTimes(
                stt=round(stt_time, 3),
                ser=round(ser_time, 3),
                llm=round(llm_time, 3) if llm_time else None,
                total=round(total_time, 3)
            ),
            audio_duration=len(audio) / sr,
            session_id=session_id
        )

    finally:
        os.unlink(tmp_path)


@app.post("/chat", response_model=Dict)
async def chat_text(
    text: str,
    emotion: Optional[str] = None,
    emotion_score: Optional[float] = None,
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID")
):
    """
    Chat with EVA using text input with conversation history.

    Headers:
        X-Session-ID: Optional session ID for conversation continuity
    """
    global pipeline

    if pipeline is None or pipeline.llm is None:
        raise HTTPException(status_code=503, detail="LLM not available")

    # Get or create session
    session_id = get_session_id(x_session_id)
    history = conversation_store.get_history(session_id)

    # Add user message to history
    user_message = ConversationMessage(
        role="user",
        content=text,
        emotion=emotion,
        emotion_score=emotion_score,
        timestamp=datetime.now().isoformat()
    )
    conversation_store.add_message(session_id, user_message)

    # Build emotion dict for prompt
    emotions = {}
    if emotion and emotion_score:
        emotions[emotion] = emotion_score
    else:
        emotions["Neutral"] = 0.5

    # Get guidelines
    guidelines = pipeline.prompt_manager.get_guidelines_for_emotion(
        emotion or "Neutral",
        emotion_score or 0.5
    )

    # Generate prompt with history
    prompt = generate_prompt_with_history(
        user_text=text,
        emotions=emotions,
        history=history[:-1],  # Exclude the message we just added
        system_context=pipeline.prompt_manager.system_context,
        guidelines=guidelines + "\n" + pipeline.prompt_manager.general_principles
    )

    try:
        start_time = time.time()
        response = pipeline.llm.generate(prompt, max_tokens=256, temperature=0.7)
        latency = time.time() - start_time

        # Add EVA response to history
        eva_message = ConversationMessage(
            role="assistant",
            content=response.text,
            timestamp=datetime.now().isoformat()
        )
        conversation_store.add_message(session_id, eva_message)

        return {
            "success": True,
            "response": response.text,
            "model": response.model,
            "processing_time": round(latency, 3),
            "session_id": session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --------------------------
# Conversation Management Endpoints
# --------------------------
@app.post("/conversation/new", response_model=Dict)
async def new_conversation():
    """Start a new conversation and return the session ID"""
    session_id = str(uuid.uuid4())
    conversation_store.get_or_create(session_id)
    return {
        "success": True,
        "session_id": session_id,
        "message": "New conversation started"
    }


@app.get("/conversation", response_model=ConversationResponse)
async def get_conversation(
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID")
):
    """Get current conversation history"""
    if not x_session_id:
        raise HTTPException(status_code=400, detail="X-Session-ID header required")

    history = conversation_store.get_history(x_session_id)

    return ConversationResponse(
        session_id=x_session_id,
        messages=history,
        message_count=len(history)
    )


@app.delete("/conversation", response_model=Dict)
async def clear_conversation(
    x_session_id: Optional[str] = Header(None, alias="X-Session-ID")
):
    """Clear conversation history for a session"""
    if not x_session_id:
        raise HTTPException(status_code=400, detail="X-Session-ID header required")

    conversation_store.clear(x_session_id)

    return {
        "success": True,
        "message": "Conversation cleared",
        "session_id": x_session_id
    }


# --------------------------
# Additional Endpoints
# --------------------------
@app.post("/transcribe", response_model=Dict)
async def transcribe_only(file: UploadFile = File(...)):
    """Transcribe audio without emotion analysis or LLM response"""
    global pipeline

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    suffix = Path(file.filename).suffix if file.filename else ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
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
    """Analyze emotions without transcription or LLM response"""
    global pipeline

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    import librosa

    suffix = Path(file.filename).suffix if file.filename else ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
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


@app.post("/reload-prompts")
async def reload_prompts():
    """Reload prompt templates from files"""
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
    import argparse

    parser = argparse.ArgumentParser(description="EVA API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║                    EVA API Server v1.1                       ║
╠══════════════════════════════════════════════════════════════╣
║  Endpoints:                                                  ║
║    POST /process         - Full pipeline with history        ║
║    POST /chat            - Text chat with history            ║
║    POST /conversation/new - Start new conversation           ║
║    GET  /conversation    - Get conversation history          ║
║    DELETE /conversation  - Clear conversation                ║
║    GET  /health          - Health check                      ║
║    GET  /docs            - Interactive API docs              ║
╚══════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "eva_api:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )


if __name__ == "__main__":
    main()
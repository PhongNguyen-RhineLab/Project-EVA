"""
TTS Module for EVA Project

Text-to-Speech engine with multiple backend support.
Primary: ElevenLabs API with Vietnamese support.
"""

from .tts_engine import (
    TTSEngine,
    TTSResponse,
    BaseTTS,
    ElevenLabsTTS,
    EdgeTTS,
    GoogleTTS
)

__all__ = [
    "TTSEngine",
    "TTSResponse",
    "BaseTTS",
    "ElevenLabsTTS",
    "EdgeTTS",
    "GoogleTTS"
]

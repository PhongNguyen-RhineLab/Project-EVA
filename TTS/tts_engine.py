"""
TTS Engine for EVA Project

Supports:
1. ElevenLabs API (premium quality, Vietnamese support)
2. gTTS (Google Text-to-Speech - free, basic)
3. Edge TTS (Microsoft Edge - free, good quality)

Primary focus: ElevenLabs with Vietnamese voice support
"""

import os
import sys
import io
import time
import tempfile
from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Union
from dataclasses import dataclass
from pathlib import Path

# Add project root to path for console import
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from console import console, Colors
except ImportError:
    # Fallback minimal console
    class Colors:
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        CYAN = '\033[96m'
        RESET = '\033[0m'

    class Console:
        def info(self, msg, indent=0): print(f"{'  '*indent}[*] {msg}")
        def success(self, msg, indent=0): print(f"{'  '*indent}[+] {msg}")
        def warning(self, msg, indent=0): print(f"{'  '*indent}[!] {msg}")
        def error(self, msg, indent=0): print(f"{'  '*indent}[-] {msg}")
        def header(self, title, width=50): print(f"\n{'='*width}\n{title}\n{'='*width}")
        def item(self, label, value, indent=1): print(f"{'  '*indent}{label}: {value}")
        def divider(self, width=50): print("-" * width)
    console = Console()


# Load environment variables from .env file if exists
def load_env():
    env_file = Path(__file__).parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().split('\n'):
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ[key.strip()] = value.strip().strip('"\'')

load_env()


@dataclass
class TTSResponse:
    """TTS response container"""
    audio_data: bytes
    format: str  # "mp3", "wav", "pcm", etc.
    sample_rate: int
    duration: Optional[float] = None
    model: str = ""
    voice: str = ""
    latency: float = 0.0
    characters_used: int = 0


# --------------------------
# Base TTS Class
# --------------------------
class BaseTTS(ABC):
    """Abstract base class for TTS backends"""

    @abstractmethod
    def synthesize(self, text: str, **kwargs) -> TTSResponse:
        """Synthesize speech from text"""
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available"""
        pass

    @abstractmethod
    def list_voices(self, language: str = None) -> List[Dict]:
        """List available voices"""
        pass


# --------------------------
# ElevenLabs TTS (Premium)
# --------------------------
class ElevenLabsTTS(BaseTTS):
    """
    ElevenLabs API - Premium quality TTS

    Features:
    - High-quality neural voices
    - Vietnamese support
    - Emotion/style control
    - Voice cloning (premium)

    Get API key: https://elevenlabs.io/

    Vietnamese voices available:
    - Use multilingual models for Vietnamese support
    """

    # Default voices (multilingual v2 supports Vietnamese)
    DEFAULT_VOICES = {
        "vi": "pNInz6obpgDQGcFmaJgB",  # Adam - good for Vietnamese
        "en": "21m00Tcm4TlvDq8ikWAM",  # Rachel - default English
    }

    # Available models
    MODELS = {
        "multilingual_v2": "eleven_multilingual_v2",  # Best for Vietnamese
        "multilingual_v1": "eleven_multilingual_v1",
        "turbo_v2": "eleven_turbo_v2",  # Faster, English only
        "monolingual": "eleven_monolingual_v1",  # English only
    }

    def __init__(
        self,
        api_key: str = None,
        model: str = "eleven_multilingual_v2",
        voice_id: str = None,
        language: str = "vi",
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        style: float = 0.0,
        use_speaker_boost: bool = True
    ):
        self.api_key = api_key or os.getenv("ELEVENLABS_API_KEY")
        self.model = model
        self.voice_id = voice_id or self.DEFAULT_VOICES.get(language, self.DEFAULT_VOICES["vi"])
        self.language = language
        self.stability = stability
        self.similarity_boost = similarity_boost
        self.style = style
        self.use_speaker_boost = use_speaker_boost
        self._client = None
        self._voices_cache = None

        if self.api_key:
            self._init_client()

    def _init_client(self):
        """Initialize ElevenLabs client"""
        try:
            from elevenlabs.client import ElevenLabs
            self._client = ElevenLabs(api_key=self.api_key)
            console.success(f"ElevenLabs initialized (model: {self.model})")
        except ImportError:
            console.warning("Install elevenlabs: pip install elevenlabs")
            self._client = None
        except Exception as e:
            console.error(f"ElevenLabs init failed: {e}")
            self._client = None

    def is_available(self) -> bool:
        return self._client is not None and self.api_key is not None

    def list_voices(self, language: str = None) -> List[Dict]:
        """List available voices"""
        if not self.is_available():
            return []

        if self._voices_cache is None:
            try:
                response = self._client.voices.get_all()
                self._voices_cache = [
                    {
                        "voice_id": voice.voice_id,
                        "name": voice.name,
                        "category": voice.category,
                        "labels": voice.labels,
                        "preview_url": voice.preview_url
                    }
                    for voice in response.voices
                ]
            except Exception as e:
                console.error(f"Failed to list voices: {e}")
                return []

        return self._voices_cache

    def synthesize(
        self,
        text: str,
        voice_id: str = None,
        model: str = None,
        output_format: str = "mp3_44100_128",
        **kwargs
    ) -> TTSResponse:
        """
        Synthesize speech from text

        Args:
            text: Text to synthesize
            voice_id: Override voice ID
            model: Override model
            output_format: Output format (mp3_44100_128, pcm_16000, etc.)

        Returns:
            TTSResponse with audio data
        """
        if not self.is_available():
            raise RuntimeError("ElevenLabs not available. Set ELEVENLABS_API_KEY.")

        start_time = time.time()

        voice = voice_id or self.voice_id
        model_id = model or self.model

        try:
            # Generate audio
            audio_generator = self._client.generate(
                text=text,
                voice=voice,
                model=model_id,
                output_format=output_format,
                voice_settings={
                    "stability": kwargs.get("stability", self.stability),
                    "similarity_boost": kwargs.get("similarity_boost", self.similarity_boost),
                    "style": kwargs.get("style", self.style),
                    "use_speaker_boost": kwargs.get("use_speaker_boost", self.use_speaker_boost)
                }
            )

            # Collect audio bytes
            audio_bytes = b"".join(audio_generator)

            latency = time.time() - start_time

            # Determine sample rate from format
            sample_rate = 44100
            if "16000" in output_format:
                sample_rate = 16000
            elif "22050" in output_format:
                sample_rate = 22050
            elif "24000" in output_format:
                sample_rate = 24000

            # Determine format
            fmt = "mp3"
            if "pcm" in output_format:
                fmt = "pcm"
            elif "wav" in output_format:
                fmt = "wav"

            return TTSResponse(
                audio_data=audio_bytes,
                format=fmt,
                sample_rate=sample_rate,
                model=model_id,
                voice=voice,
                latency=latency,
                characters_used=len(text)
            )

        except Exception as e:
            raise RuntimeError(f"ElevenLabs synthesis failed: {e}")

    def synthesize_stream(self, text: str, **kwargs):
        """
        Stream audio synthesis (for real-time playback)

        Yields audio chunks as they're generated.
        """
        if not self.is_available():
            raise RuntimeError("ElevenLabs not available")

        voice = kwargs.get("voice_id", self.voice_id)
        model_id = kwargs.get("model", self.model)

        audio_stream = self._client.generate(
            text=text,
            voice=voice,
            model=model_id,
            stream=True
        )

        for chunk in audio_stream:
            yield chunk


# --------------------------
# Edge TTS (Free, Good Quality)
# --------------------------
class EdgeTTS(BaseTTS):
    """
    Microsoft Edge TTS - Free, good quality

    Uses the same TTS engine as Microsoft Edge browser.
    Supports Vietnamese voices.

    Vietnamese voices:
    - vi-VN-HoaiMyNeural (Female)
    - vi-VN-NamMinhNeural (Male)
    """

    VIETNAMESE_VOICES = {
        "female": "vi-VN-HoaiMyNeural",
        "male": "vi-VN-NamMinhNeural"
    }

    ENGLISH_VOICES = {
        "female": "en-US-JennyNeural",
        "male": "en-US-GuyNeural"
    }

    def __init__(
        self,
        voice: str = None,
        language: str = "vi",
        rate: str = "+0%",
        volume: str = "+0%",
        pitch: str = "+0Hz"
    ):
        self.language = language
        self.rate = rate
        self.volume = volume
        self.pitch = pitch

        # Set default voice based on language
        if voice:
            self.voice = voice
        elif language == "vi":
            self.voice = self.VIETNAMESE_VOICES["female"]
        else:
            self.voice = self.ENGLISH_VOICES["female"]

        self._available = None

    def is_available(self) -> bool:
        if self._available is None:
            try:
                import edge_tts
                self._available = True
                console.success(f"Edge TTS initialized ({self.voice})")
            except ImportError:
                console.warning("Install edge-tts: pip install edge-tts")
                self._available = False

        return self._available

    def list_voices(self, language: str = None) -> List[Dict]:
        """List available voices"""
        if not self.is_available():
            return []

        import asyncio
        import edge_tts

        async def get_voices():
            voices = await edge_tts.list_voices()
            if language:
                voices = [v for v in voices if v["Locale"].startswith(language)]
            return [
                {
                    "voice_id": v["ShortName"],
                    "name": v["FriendlyName"],
                    "gender": v["Gender"],
                    "locale": v["Locale"]
                }
                for v in voices
            ]

        return asyncio.run(get_voices())

    def synthesize(
        self,
        text: str,
        voice: str = None,
        rate: str = None,
        volume: str = None,
        pitch: str = None,
        **kwargs
    ) -> TTSResponse:
        """Synthesize speech from text"""
        if not self.is_available():
            raise RuntimeError("Edge TTS not available. Install: pip install edge-tts")

        import asyncio
        import edge_tts

        start_time = time.time()

        voice = voice or self.voice
        rate = rate or self.rate
        volume = volume or self.volume
        pitch = pitch or self.pitch

        async def _synthesize():
            communicate = edge_tts.Communicate(
                text,
                voice,
                rate=rate,
                volume=volume,
                pitch=pitch
            )

            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]

            return audio_data

        audio_bytes = asyncio.run(_synthesize())
        latency = time.time() - start_time

        return TTSResponse(
            audio_data=audio_bytes,
            format="mp3",
            sample_rate=24000,
            model="edge-tts",
            voice=voice,
            latency=latency,
            characters_used=len(text)
        )


# --------------------------
# gTTS (Free, Basic)
# --------------------------
class GoogleTTS(BaseTTS):
    """
    Google Text-to-Speech (gTTS) - Free, basic quality

    Uses Google Translate's TTS API.
    Supports Vietnamese.
    """

    def __init__(self, language: str = "vi", slow: bool = False):
        self.language = language
        self.slow = slow
        self._available = None

    def is_available(self) -> bool:
        if self._available is None:
            try:
                from gtts import gTTS
                self._available = True
                console.success(f"gTTS initialized (language: {self.language})")
            except ImportError:
                console.warning("Install gtts: pip install gtts")
                self._available = False

        return self._available

    def list_voices(self, language: str = None) -> List[Dict]:
        """gTTS doesn't have voice selection - return language options"""
        return [
            {"voice_id": "vi", "name": "Vietnamese", "locale": "vi"},
            {"voice_id": "en", "name": "English", "locale": "en"},
        ]

    def synthesize(
        self,
        text: str,
        language: str = None,
        slow: bool = None,
        **kwargs
    ) -> TTSResponse:
        """Synthesize speech from text"""
        if not self.is_available():
            raise RuntimeError("gTTS not available. Install: pip install gtts")

        from gtts import gTTS

        start_time = time.time()

        lang = language or self.language
        is_slow = slow if slow is not None else self.slow

        # Generate audio
        tts = gTTS(text=text, lang=lang, slow=is_slow)

        # Save to bytes
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_bytes = audio_buffer.getvalue()

        latency = time.time() - start_time

        return TTSResponse(
            audio_data=audio_bytes,
            format="mp3",
            sample_rate=24000,
            model="gtts",
            voice=lang,
            latency=latency,
            characters_used=len(text)
        )


# --------------------------
# Unified TTS Engine
# --------------------------
class TTSEngine:
    """
    Unified TTS Engine with automatic fallback

    Priority order:
    1. Specified backend
    2. ElevenLabs (if API key available)
    3. Edge TTS (free, good quality)
    4. gTTS (free, basic)

    Usage:
        # Auto-select best available
        tts = TTSEngine(language="vi")

        # Force specific backend
        tts = TTSEngine(backend="elevenlabs")

        # Synthesize
        response = tts.synthesize("Xin chào!")
        with open("output.mp3", "wb") as f:
            f.write(response.audio_data)
    """

    BACKENDS = {
        "elevenlabs": ElevenLabsTTS,
        "edge": EdgeTTS,
        "gtts": GoogleTTS
    }

    def __init__(
        self,
        backend: str = None,
        language: str = "vi",
        voice: str = None,
        api_key: str = None,
        **kwargs
    ):
        """
        Initialize TTS Engine

        Args:
            backend: Specific backend to use (elevenlabs, edge, gtts)
            language: Language code (vi, en, etc.)
            voice: Voice ID or name
            api_key: API key (for ElevenLabs)
            **kwargs: Additional backend-specific arguments
        """
        self.backend_name = backend
        self.language = language
        self._backend = None

        console.header("Initializing TTS Engine")

        if backend:
            self._init_specific_backend(backend, language, voice, api_key, **kwargs)
        else:
            self._init_auto_backend(language, voice, api_key, **kwargs)

        if self._backend and self._backend.is_available():
            console.success(f"TTS Engine ready ({self.backend_name})")
        else:
            console.warning("No TTS backend available!")
            self._print_setup_help()

        console.divider()
        print()

    def _init_specific_backend(
        self,
        backend: str,
        language: str,
        voice: str,
        api_key: str,
        **kwargs
    ):
        """Initialize a specific backend"""
        if backend not in self.BACKENDS:
            raise ValueError(f"Unknown backend: {backend}. Available: {list(self.BACKENDS.keys())}")

        backend_class = self.BACKENDS[backend]

        init_kwargs = {"language": language, **kwargs}

        if backend == "elevenlabs":
            if api_key:
                init_kwargs["api_key"] = api_key
            if voice:
                init_kwargs["voice_id"] = voice
        elif voice:
            init_kwargs["voice"] = voice

        self._backend = backend_class(**init_kwargs)
        self.backend_name = backend

    def _init_auto_backend(
        self,
        language: str,
        voice: str,
        api_key: str,
        **kwargs
    ):
        """Auto-detect and initialize best available backend"""
        console.info("Auto-detecting available TTS backends...")

        # Try ElevenLabs first (best quality)
        if api_key or os.getenv("ELEVENLABS_API_KEY"):
            try:
                init_kwargs = {"language": language, **kwargs}
                if api_key:
                    init_kwargs["api_key"] = api_key
                if voice:
                    init_kwargs["voice_id"] = voice

                self._backend = ElevenLabsTTS(**init_kwargs)
                if self._backend.is_available():
                    self.backend_name = "elevenlabs"
                    console.item("elevenlabs", f"{Colors.GREEN}available{Colors.RESET}")
                    return
            except Exception:
                console.item("elevenlabs", "not available")

        # Try Edge TTS (free, good quality)
        try:
            init_kwargs = {"language": language}
            if voice:
                init_kwargs["voice"] = voice

            self._backend = EdgeTTS(**init_kwargs)
            if self._backend.is_available():
                self.backend_name = "edge"
                console.item("edge", f"{Colors.GREEN}available{Colors.RESET}")
                return
        except Exception:
            console.item("edge", "not available")

        # Try gTTS (free, basic)
        try:
            self._backend = GoogleTTS(language=language)
            if self._backend.is_available():
                self.backend_name = "gtts"
                console.item("gtts", f"{Colors.GREEN}available{Colors.RESET}")
                return
        except Exception:
            console.item("gtts", "not available")

        self._backend = None
        self.backend_name = None

    def _print_setup_help(self):
        """Print help for setting up backends"""
        print("\nSetup instructions:")
        print("\n[Option 1] ElevenLabs (Recommended - Best Quality)")
        print("   1. Sign up: https://elevenlabs.io/")
        print("   2. Get API key from your profile")
        print("   3. Set: export ELEVENLABS_API_KEY=your_key")
        print("   4. Or add to .env file: ELEVENLABS_API_KEY=your_key")

        print("\n[Option 2] Edge TTS (Free, Good Quality)")
        print("   1. Install: pip install edge-tts")
        print("   2. No API key required")

        print("\n[Option 3] gTTS (Free, Basic)")
        print("   1. Install: pip install gtts")
        print("   2. No API key required")

    def is_available(self) -> bool:
        """Check if any backend is available"""
        return self._backend is not None and self._backend.is_available()

    def synthesize(
        self,
        text: str,
        **kwargs
    ) -> TTSResponse:
        """
        Synthesize speech from text

        Args:
            text: Text to synthesize
            **kwargs: Backend-specific options

        Returns:
            TTSResponse with audio data
        """
        if not self.is_available():
            raise RuntimeError("No TTS backend available. See setup instructions.")

        return self._backend.synthesize(text, **kwargs)

    def synthesize_to_file(
        self,
        text: str,
        output_path: str,
        **kwargs
    ) -> TTSResponse:
        """
        Synthesize speech and save to file

        Args:
            text: Text to synthesize
            output_path: Output file path
            **kwargs: Backend-specific options

        Returns:
            TTSResponse with audio data
        """
        response = self.synthesize(text, **kwargs)

        with open(output_path, "wb") as f:
            f.write(response.audio_data)

        console.success(f"Audio saved to {output_path}")

        return response

    def list_voices(self, language: str = None) -> List[Dict]:
        """List available voices for current backend"""
        if not self.is_available():
            return []

        return self._backend.list_voices(language or self.language)

    def list_backends(self) -> Dict[str, bool]:
        """List all backends and their availability"""
        status = {}

        # Check ElevenLabs
        try:
            backend = ElevenLabsTTS()
            status["elevenlabs"] = backend.is_available()
        except:
            status["elevenlabs"] = False

        # Check Edge TTS
        try:
            backend = EdgeTTS()
            status["edge"] = backend.is_available()
        except:
            status["edge"] = False

        # Check gTTS
        try:
            backend = GoogleTTS()
            status["gtts"] = backend.is_available()
        except:
            status["gtts"] = False

        return status


# --------------------------
# Test Function
# --------------------------
def test_tts():
    """Test TTS engine"""
    console.header("TTS Engine Test")

    # Check available backends
    console.info("Checking backends...")
    engine = TTSEngine(language="vi")

    status = engine.list_backends()
    for backend, available in status.items():
        status_str = f"{Colors.GREEN}yes{Colors.RESET}" if available else f"{Colors.RED}no{Colors.RESET}"
        console.item(backend, status_str)

    if not engine.is_available():
        console.error("No backend available. Follow setup instructions above.")
        return

    # List voices
    console.info(f"Available voices for {engine.backend_name}:")
    voices = engine.list_voices()
    for voice in voices[:5]:  # Show first 5
        console.item(voice.get("name", voice.get("voice_id")), voice.get("voice_id", ""))

    # Test synthesis
    console.info(f"Testing synthesis with {engine.backend_name}...")

    test_text = "Xin chào! Tôi là EVA, trợ lý giọng nói thấu cảm của bạn."

    try:
        response = engine.synthesize(test_text)

        console.success("Synthesis complete!")
        console.item("Format", response.format)
        console.item("Sample Rate", f"{response.sample_rate} Hz")
        console.item("Size", f"{len(response.audio_data)} bytes")
        console.item("Latency", f"{response.latency:.2f}s")
        console.item("Characters", response.characters_used)

        # Save test file
        test_file = PROJECT_ROOT / "test_tts_output.mp3"
        with open(test_file, "wb") as f:
            f.write(response.audio_data)

        console.success(f"Test audio saved to: {test_file}")

    except Exception as e:
        console.error(f"Error: {e}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_tts()
    elif len(sys.argv) > 1 and sys.argv[1] == "--list-voices":
        engine = TTSEngine(language="vi")
        voices = engine.list_voices()
        console.header("Available Voices")
        for voice in voices:
            print(f"  {voice.get('name', 'N/A'):30s} | {voice.get('voice_id', 'N/A')}")
    else:
        print("TTS Engine for EVA Project")
        print("=" * 50)
        print("\nUsage:")
        print("  python tts_engine.py --test          # Test TTS")
        print("  python tts_engine.py --list-voices   # List voices")
        print("\nBackends available:")
        print("  - elevenlabs (Premium, best quality)")
        print("  - edge (Free, good quality)")
        print("  - gtts (Free, basic)")
        print("\nSetup (ElevenLabs):")
        print("  1. Sign up at https://elevenlabs.io/")
        print("  2. Get API key from profile")
        print("  3. Create .env file with: ELEVENLABS_API_KEY=your_key")
        print("  4. Run: python tts_engine.py --test")
        print("\nVietnamese voices:")
        print("  - ElevenLabs: Use multilingual_v2 model (auto)")
        print("  - Edge TTS: vi-VN-HoaiMyNeural, vi-VN-NamMinhNeural")

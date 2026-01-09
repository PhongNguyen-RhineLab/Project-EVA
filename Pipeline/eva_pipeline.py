"""
EVA Pipeline - Speech-to-Text + Speech Emotion Recognition + LLM

Unified pipeline for processing voice input and generating empathic responses.
"""

import sys
import time
import numpy as np
import librosa
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Project root
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
        def header(self, title, width=60): print(f"\n{'='*width}\n{title.center(width)}\n{'='*width}")
        def subheader(self, title, width=60): print(f"\n{'-'*width}\n{title}\n{'-'*width}")
        def item(self, label, value, indent=1): print(f"{'  '*indent}{label}: {value}")
        def divider(self, width=60, char="-"): print(char * width)
        def emotion(self, name, score, width=20):
            bar = "█" * int(width * score) + "░" * (width - int(width * score))
            print(f"  {name:12s} [{bar}] {score*100:5.1f}%")
    console = Console()


# --------------------------
# Data Classes
# --------------------------
@dataclass
class STTResult:
    """Speech-to-Text result"""
    text: str
    language: str
    confidence: Optional[float]
    segments: List[Dict]
    processing_time: float


@dataclass
class SERResult:
    """Speech Emotion Recognition result"""
    emotions: Dict[str, float]
    dominant_emotions: Dict[str, float]
    latent_vector: Optional[np.ndarray]
    processing_time: float


@dataclass
class TTSResult:
    """Text-to-Speech result"""
    audio_data: bytes
    format: str
    sample_rate: int
    duration: Optional[float]
    processing_time: float


@dataclass
class PipelineResult:
    """Complete pipeline result"""
    # STT
    text: str
    stt_result: STTResult
    stt_confidence: Optional[float]

    # SER
    emotions: Dict[str, float]
    dominant_emotions: Dict[str, float]
    ser_result: SERResult

    # LLM
    llm_prompt: str
    llm_response: Optional[str]
    llm_result: Optional[object]

    # TTS
    tts_result: Optional[TTSResult] = None
    audio_response: Optional[bytes] = None

    # Meta
    total_processing_time: float = 0.0


# --------------------------
# Prompt Manager
# --------------------------
class PromptManager:
    """Manages LLM prompts for empathic responses"""

    def __init__(self, prompts_dir: Path = None):
        if prompts_dir is None:
            prompts_dir = PROJECT_ROOT / "prompts"

        self.prompts_dir = Path(prompts_dir)
        self.guidelines = {}
        self.system_context = ""
        self.general_principles = ""

        self._load_prompts()

    def _load_prompts(self):
        """Load prompt templates from files"""
        # Load system context
        system_file = self.prompts_dir / "system_context.txt"
        if system_file.exists():
            self.system_context = system_file.read_text().strip()
        else:
            self.system_context = """You are EVA, an empathic voice assistant designed to provide emotional support.
Your role is to:
- Listen actively and respond with genuine empathy
- Validate the user's feelings
- Provide supportive, non-judgmental responses
- Help users process their emotions constructively"""

        # Load emotion-specific guidelines
        guidelines_dir = self.prompts_dir / "guidelines"
        if guidelines_dir.exists():
            for file in guidelines_dir.glob("*.txt"):
                key = file.stem.upper()
                self.guidelines[key] = file.read_text().strip()
        else:
            # Default guidelines
            self.guidelines = {
                'DEFAULT': "Respond with empathy and understanding.",
                'SAD_FEARFUL': "Show extra compassion. Validate their feelings.",
                'ANGRY_DISGUST': "Stay calm. Acknowledge their frustration.",
                'HAPPY_SURPRISED': "Share in their positive emotions."
            }
        self.general_principles = "- Respond with empathy and understanding"

    def get_guidelines_for_emotion(self, emotion: str, intensity: float) -> str:
        """Get appropriate guidelines based on detected emotion"""
        if emotion in ["Sad", "Fearful"] and intensity > 0.5:
            return self.guidelines.get('SAD_FEARFUL', self.guidelines.get('DEFAULT', ''))
        elif emotion in ["Angry", "Disgust"] and intensity > 0.5:
            return self.guidelines.get('ANGRY_DISGUST', self.guidelines.get('DEFAULT', ''))
        elif emotion in ["Happy", "Surprised"] and intensity > 0.5:
            return self.guidelines.get('HAPPY_SURPRISED', self.guidelines.get('DEFAULT', ''))
        else:
            return self.guidelines.get('DEFAULT', '')

    def reload(self):
        """Reload prompts from files"""
        self.guidelines = {}
        self._load_prompts()
        console.info("Prompts reloaded")


# --------------------------
# SER Module Wrapper
# --------------------------
class SERModule:
    """Speech Emotion Recognition using trained Beta-VAE model"""

    EMOTION_LABELS = [
        "Neutral", "Calm", "Happy", "Sad",
        "Angry", "Fearful", "Disgust", "Surprised"
    ]

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "auto",
        sr: int = 16000,
        n_mels: int = 128,
        duration: int = 3,
        hop_length: int = 512,
        n_fft: int = 2048
    ):
        import torch

        self.sr = sr
        self.n_mels = n_mels
        self.duration = duration
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.target_frames = int(sr * duration / hop_length) + 1

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load model
        console.info(f"Loading SER model from {checkpoint_path}")
        self._load_model(checkpoint_path)
        console.success(f"SER ready on {self.device}")

    def _load_model(self, checkpoint_path: str):
        """Load trained Beta-VAE model"""
        import torch

        # Try different import paths
        try:
            from VAE.model import BetaVAE_SER
        except ImportError:
            try:
                from model import BetaVAE_SER
            except ImportError:
                sys.path.insert(0, str(PROJECT_ROOT / "VAE"))
                from VAE.model import BetaVAE_SER

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        config = checkpoint['config']

        self.model = BetaVAE_SER(
            n_mels=config.get('n_mels', 128),
            n_emotions=config.get('n_emotions', 8),
            latent_dim=config.get('latent_dim', 64)
        ).to(self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        console.success(f"SER model loaded (epoch {checkpoint['epoch']}, F1: {checkpoint.get('f1_micro', 0):.4f})")

    def _extract_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Extract mel spectrogram from audio"""
        # Pad or trim to target duration
        target_samples = self.sr * self.duration
        if len(audio) > target_samples:
            audio = audio[:target_samples]
        else:
            audio = np.pad(audio, (0, target_samples - len(audio)), mode='constant')

        # Extract mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length
        )

        # Convert to dB and normalize to [0, 1]
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-6)

        return mel_spec_norm

    def predict(self, audio: np.ndarray, sr: int = 16000, threshold: float = 0.5) -> SERResult:
        """Predict emotions from audio"""
        import torch

        start_time = time.time()

        # Resample if needed
        if sr != self.sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)

        # Extract features
        mel_spec = self._extract_mel_spectrogram(audio)

        # Prepare input tensor
        x = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            # Model returns: x_recon, y_pred, mu, log_var
            _, y_pred, mu, _ = self.model(x)
            probs = y_pred.cpu().numpy()[0]
            latent = mu.cpu().numpy()[0]

        # Build emotions dict
        emotions = {label: float(prob) for label, prob in zip(self.EMOTION_LABELS, probs)}

        # Get dominant emotions
        dominant = {k: v for k, v in emotions.items() if v >= threshold}

        processing_time = time.time() - start_time

        return SERResult(
            emotions=emotions,
            dominant_emotions=dominant,
            latent_vector=latent,
            processing_time=processing_time
        )

    def predict_file(self, audio_path: str, threshold: float = 0.5) -> SERResult:
        """Predict emotions from audio file"""
        audio, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        return self.predict(audio, sr, threshold)


# --------------------------
# STT Module Wrapper
# --------------------------
class STTModule:
    """Speech-to-Text wrapper for pipeline integration"""

    def __init__(
        self,
        backend: str = "whisper",
        model_size: str = "base",
        language: str = "vi",
        device: str = "auto"
    ):
        try:
            from STT.stt_engine import STTEngine
        except ImportError:
            try:
                from stt_engine import STTEngine
            except ImportError:
                sys.path.insert(0, str(PROJECT_ROOT / "STT"))
                from STT.stt_engine import STTEngine

        console.info(f"Initializing STT module ({backend}/{model_size})")

        self.engine = STTEngine(
            backend=backend,
            whisper_model=model_size,
            language=language,
            device=device
        )
        self.language = language

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> STTResult:
        """Transcribe audio array"""
        start_time = time.time()
        result = self.engine.transcribe_array(audio, sr=sr)
        processing_time = time.time() - start_time

        return STTResult(
            text=result['text'],
            language=result['language'],
            confidence=result['confidence'],
            segments=result.get('segments', []),
            processing_time=processing_time
        )

    def transcribe_file(self, audio_path: str) -> STTResult:
        """Transcribe audio file"""
        start_time = time.time()
        result = self.engine.transcribe(audio_path)
        processing_time = time.time() - start_time

        return STTResult(
            text=result['text'],
            language=result['language'],
            confidence=result['confidence'],
            segments=result.get('segments', []),
            processing_time=processing_time
        )


# --------------------------
# Main Pipeline
# --------------------------
class EVAPipeline:
    """
    Main EVA Pipeline - Orchestrates STT + SER for empathic responses

    Usage:
        pipeline = EVAPipeline(
            ser_checkpoint="checkpoints/best_model.pth",
            stt_model="base",
            language="vi"
        )

        result = pipeline.process("audio.wav")
        print(result.llm_prompt)
    """

    def __init__(
        self,
        ser_checkpoint: str,
        stt_backend: str = "whisper",
        stt_model: str = "base",
        language: str = "vi",
        device: str = "auto",
        emotion_threshold: float = 0.5,
        parallel: bool = True,
        prompts_dir: Path = None,
        llm_backend: str = None,
        llm_model: str = None,
        llm_api_key: str = None,
        enable_llm: bool = True,
        tts_backend: str = None,
        tts_voice: str = None,
        tts_api_key: str = None,
        enable_tts: bool = True
    ):
        self.emotion_threshold = emotion_threshold
        self.parallel = parallel
        self.language = language
        self.enable_llm = enable_llm
        self.enable_tts = enable_tts

        console.header("Initializing EVA Pipeline")

        # Initialize prompt manager
        self.prompt_manager = PromptManager(prompts_dir)

        # Initialize modules
        self.stt = STTModule(
            backend=stt_backend,
            model_size=stt_model,
            language=language,
            device=device
        )

        self.ser = SERModule(
            checkpoint_path=ser_checkpoint,
            device=device
        )

        # Initialize LLM
        self.llm = None
        if enable_llm:
            self._init_llm(llm_backend, llm_model, llm_api_key)

        # Initialize TTS
        self.tts = None
        if enable_tts:
            self._init_tts(tts_backend, tts_voice, tts_api_key)

        console.divider()
        console.success("EVA Pipeline ready!")
        console.divider()
        print()

    def _init_llm(self, backend: str, model: str, api_key: str):
        """Initialize LLM backend"""
        try:
            from LLM.llm_engine import LLMEngine
        except ImportError:
            try:
                from llm_engine import LLMEngine
            except ImportError:
                sys.path.insert(0, str(PROJECT_ROOT / "LLM"))
                from LLM.llm_engine import LLMEngine

        self.llm = LLMEngine(
            backend=backend,
            model=model,
            api_key=api_key
        )

    def _init_tts(self, backend: str, voice: str, api_key: str):
        """Initialize TTS backend"""
        try:
            from TTS.tts_engine import TTSEngine
        except ImportError:
            try:
                from tts_engine import TTSEngine
            except ImportError:
                sys.path.insert(0, str(PROJECT_ROOT / "TTS"))
                from TTS.tts_engine import TTSEngine

        self.tts = TTSEngine(
            backend=backend,
            language=self.language,
            voice=voice,
            api_key=api_key
        )

    def process(
        self,
        audio_path: str,
        generate_response: bool = True,
        generate_audio: bool = True
    ) -> PipelineResult:
        """
        Process audio file through full pipeline

        Args:
            audio_path: Path to audio file
            generate_response: Whether to generate LLM response
            generate_audio: Whether to generate TTS audio response

        Returns:
            PipelineResult with all outputs
        """
        start_time = time.time()

        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)

        # Run STT and SER
        if self.parallel:
            stt_result, ser_result = self._process_parallel(audio, sr)
        else:
            stt_result = self.stt.transcribe(audio, sr)
            ser_result = self.ser.predict(audio, sr, self.emotion_threshold)

        # Get primary emotion
        primary_emotion = max(ser_result.emotions.items(), key=lambda x: x[1])

        # Build LLM prompt
        llm_prompt = self._build_prompt(
            stt_result.text,
            ser_result.emotions,
            primary_emotion
        )

        # Generate LLM response
        llm_response = None
        llm_result = None

        if generate_response and self.llm and self.llm.is_available():
            try:
                llm_result = self.llm.generate(llm_prompt, max_tokens=512, temperature=0.7)
                llm_response = llm_result.text
            except Exception as e:
                console.warning(f"LLM generation failed: {e}")

        # Generate TTS audio
        tts_result = None
        audio_response = None

        if generate_audio and llm_response and self.tts and self.tts.is_available():
            try:
                tts_start = time.time()
                tts_response = self.tts.synthesize(llm_response)
                tts_time = time.time() - tts_start

                audio_duration = len(tts_response.audio_data) / (
                            tts_response.sample_rate * 2) if tts_response.format == 'pcm' else None

                tts_result = TTSResult(
                    audio_data=tts_response.audio_data,
                    format=tts_response.format,
                    sample_rate=tts_response.sample_rate,
                    duration=audio_duration,
                    processing_time=tts_time
                )
                audio_response = tts_response.audio_data
            except Exception as e:
                console.warning(f"TTS generation failed: {e}")

        total_time = time.time() - start_time

        result = PipelineResult(
            text=stt_result.text,
            stt_result=stt_result,
            stt_confidence=stt_result.confidence,
            emotions=ser_result.emotions,
            dominant_emotions=ser_result.dominant_emotions,
            ser_result=ser_result,
            llm_prompt=llm_prompt,
            llm_response=llm_response,
            llm_result=llm_result,
            tts_result=tts_result,
            audio_response=audio_response,
            total_processing_time=total_time
        )

        # Print results
        self._print_result(result)

        return result

    def _process_parallel(self, audio: np.ndarray, sr: int) -> Tuple[STTResult, SERResult]:
        """Run STT and SER in parallel"""
        with ThreadPoolExecutor(max_workers=2) as executor:
            stt_future = executor.submit(self.stt.transcribe, audio, sr)
            ser_future = executor.submit(self.ser.predict, audio, sr, self.emotion_threshold)

            stt_result = stt_future.result()
            ser_result = ser_future.result()

        return stt_result, ser_result

    def _build_prompt(
        self,
        text: str,
        emotions: Dict[str, float],
        primary_emotion: Tuple[str, float]
    ) -> str:
        """Build LLM prompt with context"""
        emotion_name, emotion_score = primary_emotion

        # Get emotion-specific guidelines
        guidelines = self.prompt_manager.get_guidelines_for_emotion(
            emotion_name, emotion_score
        )

        # Format emotion scores
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        emotion_str = ", ".join([f"{e}: {s*100:.1f}%" for e, s in sorted_emotions[:3]])

        prompt = f"""{self.prompt_manager.system_context}

{guidelines}

---
User's emotional state: {emotion_name} ({emotion_score*100:.1f}%)
All emotions: {emotion_str}
User's message: {text}
---

Provide a warm, empathic response:"""

        return prompt

    def _print_result(self, result: PipelineResult):
        """Print formatted result"""
        console.subheader("Processing Result")

        # Transcription
        console.info("Transcription:")
        print(f"   {result.text}")
        if result.stt_confidence:
            console.item("Confidence", f"{result.stt_confidence:.1%}")

        # Emotions
        print()
        console.info("Emotions:")
        sorted_emotions = sorted(
            result.emotions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for emotion, prob in sorted_emotions[:4]:
            console.emotion(emotion, prob)

        # LLM Response
        if result.llm_response:
            print()
            console.info("EVA Response:")
            response_lines = result.llm_response.strip().split('\n')
            for line in response_lines:
                print(f"   {line}")

        # Timing
        print()
        console.info("Timing:")
        console.item("STT", f"{result.stt_result.processing_time:.2f}s")
        console.item("SER", f"{result.ser_result.processing_time:.2f}s")
        if result.llm_result:
            console.item("LLM", f"{result.llm_result.latency:.2f}s ({result.llm_result.model})")
        if result.tts_result:
            console.item("TTS", f"{result.tts_result.processing_time:.2f}s ({result.tts_result.format})")
        console.item("Total", f"{result.total_processing_time:.2f}s")

        console.divider()
        print()


# --------------------------
# Quick Test
# --------------------------
def test_pipeline():
    """Quick test of the pipeline"""
    console.header("EVA Pipeline Test")

    # Check for required files
    checkpoint = PROJECT_ROOT / "checkpoints" / "best_model.pth"
    test_audio = PROJECT_ROOT / "test_audio.wav"

    if not checkpoint.exists():
        console.error(f"Checkpoint not found: {checkpoint}")
        console.info("Train a model first or update the path")
        return

    if not test_audio.exists():
        console.error(f"Test audio not found: {test_audio}")
        return

    # Initialize pipeline
    pipeline = EVAPipeline(
        ser_checkpoint=str(checkpoint),
        stt_model="base",
        language="vi",
        parallel=True,
        enable_llm=True
    )

    # Process
    result = pipeline.process(str(test_audio))

    # Show results
    if result.llm_response:
        console.header("EVA Response")
        print(result.llm_response)
    else:
        console.header("Generated LLM Prompt (no LLM available)")
        print(result.llm_prompt)

    console.success("Pipeline test complete!")


# --------------------------
# CLI Interface
# --------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EVA Pipeline - STT + SER + LLM + TTS")
    parser.add_argument("audio", nargs="?", default=None, help="Audio file to process")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth", help="SER model checkpoint")
    parser.add_argument("--stt-model", default="base", help="Whisper model size")
    parser.add_argument("--language", default="vi", help="Language code")
    parser.add_argument("--sequential", action="store_true", help="Run sequentially instead of parallel")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM response generation")
    parser.add_argument("--llm-backend", default=None, help="LLM backend (groq, gemini, ollama)")
    parser.add_argument("--llm-model", default=None, help="LLM model name")
    parser.add_argument("--no-tts", action="store_true", help="Disable TTS audio generation")
    parser.add_argument("--tts-backend", default=None, help="TTS backend (elevenlabs, edge, gtts)")
    parser.add_argument("--tts-voice", default=None, help="TTS voice ID")
    parser.add_argument("--output-audio", default=None, help="Save TTS output to file")
    parser.add_argument("--test", action="store_true", help="Run test")

    args = parser.parse_args()

    if args.test:
        test_pipeline()
    elif args.audio:
        pipeline = EVAPipeline(
            ser_checkpoint=args.checkpoint,
            stt_model=args.stt_model,
            language=args.language,
            parallel=not args.sequential,
            enable_llm=not args.no_llm,
            llm_backend=args.llm_backend,
            llm_model=args.llm_model,
            enable_tts=not args.no_tts,
            tts_backend=args.tts_backend,
            tts_voice=args.tts_voice
        )
        result = pipeline.process(args.audio, generate_audio=not args.no_tts)

        if result.llm_response:
            console.header("EVA Response")
            print(result.llm_response)

        # Save TTS output if requested
        if result.audio_response and args.output_audio:
            with open(args.output_audio, "wb") as f:
                f.write(result.audio_response)
            console.success(f"Audio saved to: {args.output_audio}")
    else:
        print("EVA Pipeline - Empathic Voice Assistant")
        print("=" * 50)
        print("\nUsage:")
        print("  python eva_pipeline.py audio.wav          # Process audio file")
        print("  python eva_pipeline.py --test             # Run test")
        print("  python eva_pipeline.py --help             # Show all options")
        print("\nLLM Options:")
        print("  --llm-backend groq                        # Use Groq API")
        print("  --llm-backend ollama                      # Use local Ollama")
        print("  --no-llm                                  # Disable LLM")
        print("\nTTS Options:")
        print("  --tts-backend elevenlabs                  # Use ElevenLabs (best)")
        print("  --tts-backend edge                        # Use Edge TTS (free)")
        print("  --no-tts                                  # Disable TTS")
        print("  --output-audio response.mp3               # Save audio to file")
        print("\nSetup:")
        print("  1. LLM: Get API key from https://console.groq.com/keys")
        print("  2. TTS: Get API key from https://elevenlabs.io/")
        print("  3. Create .env file with API keys")
        print("  4. Run: python eva_pipeline.py --test")
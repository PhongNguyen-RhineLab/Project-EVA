"""
EVA Pipeline - Orchestrates STT + SER for Empathic Response Generation

Pipeline flow:
    Audio ─┬─→ STT ──────→ text ─────────┐
           │                              ├─→ LLM Prompt → Response
           └─→ SER (VAE) → emotions ─────┘
"""

import torch
import numpy as np
import librosa
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import warnings
import sys

warnings.filterwarnings('ignore')

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"
sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------
# Data Classes
# --------------------------
@dataclass
class STTResult:
    """Speech-to-Text result"""
    text: str
    language: str
    confidence: Optional[float]
    segments: List[dict]
    processing_time: float


@dataclass
class SERResult:
    """Speech Emotion Recognition result"""
    emotions: Dict[str, float]  # All emotion probabilities
    dominant_emotions: Dict[str, float]  # Emotions above threshold
    latent_vector: np.ndarray
    processing_time: float


@dataclass
class LLMResult:
    """LLM response result"""
    response: str
    model: str
    latency: float
    tokens_used: Optional[int] = None


@dataclass
class PipelineResult:
    """Combined pipeline result"""
    text: str
    emotions: Dict[str, float]
    dominant_emotions: Dict[str, float]
    llm_prompt: str
    llm_response: Optional[str]  # Generated response
    stt_confidence: Optional[float]
    total_processing_time: float

    # Individual results for debugging
    stt_result: STTResult
    ser_result: SERResult
    llm_result: Optional[LLMResult] = None


# --------------------------
# Prompt Manager
# --------------------------
class PromptManager:
    """
    Manages prompt templates loaded from external files

    Files:
        prompts/system_context.txt - System context for LLM
        prompts/response_guidelines.txt - Emotion-specific guidelines
    """

    def __init__(self, prompts_dir: Path = None):
        """
        Initialize prompt manager

        Args:
            prompts_dir: Directory containing prompt files
        """
        self.prompts_dir = prompts_dir or PROMPTS_DIR
        self.system_context = ""
        self.guidelines = {}
        self.general_principles = ""

        self._load_prompts()

    def _load_prompts(self):
        """Load prompts from files"""
        # Load system context
        system_file = self.prompts_dir / "system_context.txt"
        if system_file.exists():
            self.system_context = system_file.read_text(encoding='utf-8').strip()
            print(f"✅ Loaded system context from {system_file.name}")
        else:
            print(f"⚠️  System context file not found: {system_file}")
            self.system_context = "You are EVA, an empathic voice assistant."

        # Load response guidelines
        guidelines_file = self.prompts_dir / "response_guidelines.txt"
        if guidelines_file.exists():
            self._parse_guidelines(guidelines_file.read_text(encoding='utf-8'))
            print(f"✅ Loaded response guidelines from {guidelines_file.name}")
        else:
            print(f"⚠️  Guidelines file not found: {guidelines_file}")
            self._set_default_guidelines()

    def _parse_guidelines(self, content: str):
        """Parse guidelines file into sections"""
        current_section = None
        current_lines = []

        for line in content.split('\n'):
            line = line.strip()

            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Check for section header
            if line.startswith('[') and line.endswith(']'):
                # Save previous section
                if current_section and current_lines:
                    self.guidelines[current_section] = '\n'.join(current_lines)

                # Start new section
                current_section = line[1:-1]  # Remove brackets
                current_lines = []
            else:
                current_lines.append(line)

        # Save last section
        if current_section and current_lines:
            self.guidelines[current_section] = '\n'.join(current_lines)

        # Extract general principles
        self.general_principles = self.guidelines.get('GENERAL_PRINCIPLES', '')

    def _set_default_guidelines(self):
        """Set default guidelines if file not found"""
        self.guidelines = {
            'SAD_FEARFUL': "- Use a gentle, supportive tone\n- Acknowledge their feelings first",
            'ANGRY_DISGUST': "- Stay calm and non-judgmental\n- Validate their frustration",
            'HAPPY_SURPRISED': "- Match their positive energy\n- Share in their joy/excitement",
            'DEFAULT': "- Be warm and conversational\n- Show genuine interest"
        }
        self.general_principles = "- Respond with empathy and understanding"

    def get_guidelines_for_emotion(self, emotion: str, intensity: float) -> str:
        """
        Get appropriate guidelines based on detected emotion

        Args:
            emotion: Primary emotion detected
            intensity: Emotion intensity (0-1)

        Returns:
            Guidelines string
        """
        # Map emotions to guideline categories
        if emotion in ["Sad", "Fearful"] and intensity > 0.5:
            return self.guidelines.get('SAD_FEARFUL', self.guidelines.get('DEFAULT', ''))
        elif emotion in ["Angry", "Disgust"] and intensity > 0.5:
            return self.guidelines.get('ANGRY_DISGUST', self.guidelines.get('DEFAULT', ''))
        elif emotion in ["Happy", "Surprised"] and intensity > 0.5:
            return self.guidelines.get('HAPPY_SURPRISED', self.guidelines.get('DEFAULT', ''))
        else:
            return self.guidelines.get('DEFAULT', '')

    def reload(self):
        """Reload prompts from files (useful for hot-reloading)"""
        self.guidelines = {}
        self._load_prompts()
        print("🔄 Prompts reloaded")


# --------------------------
# SER Module Wrapper
# --------------------------
class SERModule:
    """
    Speech Emotion Recognition using trained Beta-VAE model
    """

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
        """
        Initialize SER module

        Args:
            checkpoint_path: Path to trained model checkpoint
            device: "cuda", "cpu", or "auto"
            sr: Sample rate for audio processing
            n_mels: Number of mel frequency bins
            duration: Audio duration in seconds
            hop_length: Hop length for spectrogram
            n_fft: FFT window size
        """
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Audio params
        self.sr = sr
        self.n_mels = n_mels
        self.duration = duration
        self.hop_length = hop_length
        self.n_fft = n_fft

        print(f"🎭 Initializing SER module on {self.device}")

        # Load model
        self._load_model(checkpoint_path)

    def _load_model(self, checkpoint_path: str):
        """Load trained Beta-VAE model"""
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

        print(f"✅ SER model loaded (epoch {checkpoint['epoch']}, F1: {checkpoint.get('f1_micro', 'N/A'):.4f})")

    def _preprocess_audio(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        """Convert audio to mel spectrogram tensor"""
        # Resample if needed
        if sr != self.sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)

        # Pad/truncate to fixed length
        max_len = self.duration * self.sr
        if len(audio) > max_len:
            audio = audio[:max_len]
        else:
            audio = np.pad(audio, (0, max_len - len(audio)), mode="constant")

        # Compute mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Normalize to [0, 1]
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)

        # Convert to tensor: (1, 1, n_mels, time)
        mel_tensor = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        return mel_tensor

    def predict(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        threshold: float = 0.5
    ) -> SERResult:
        """
        Predict emotions from audio array

        Args:
            audio: Audio samples (mono, float)
            sr: Sample rate
            threshold: Threshold for dominant emotions

        Returns:
            SERResult with emotion predictions
        """
        start_time = time.time()

        # Preprocess
        mel_tensor = self._preprocess_audio(audio, sr).to(self.device)

        # Inference
        with torch.no_grad():
            _, y_pred, mu, _ = self.model(mel_tensor)

        # Convert to numpy
        probs = y_pred.cpu().numpy()[0]
        latent = mu.cpu().numpy()[0]

        # Create emotion dictionary
        emotions = {
            label: float(prob)
            for label, prob in zip(self.EMOTION_LABELS, probs)
        }

        # Get dominant emotions (above threshold)
        dominant = {
            label: prob
            for label, prob in emotions.items()
            if prob > threshold
        }

        processing_time = time.time() - start_time

        return SERResult(
            emotions=emotions,
            dominant_emotions=dominant,
            latent_vector=latent,
            processing_time=processing_time
        )

    def predict_from_file(self, audio_path: str, threshold: float = 0.5) -> SERResult:
        """Predict emotions from audio file"""
        audio, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        return self.predict(audio, sr, threshold)


# --------------------------
# STT Module Wrapper
# --------------------------
class STTModule:
    """
    Speech-to-Text wrapper for pipeline integration
    """

    def __init__(
        self,
        backend: str = "whisper",
        model_size: str = "base",
        language: str = "vi",
        device: str = "auto"
    ):
        """
        Initialize STT module

        Args:
            backend: "whisper" or "vosk"
            model_size: Whisper model size
            language: Language code
            device: Device for computation
        """
        # Try different import paths
        try:
            from STT.stt_engine import STTEngine
        except ImportError:
            try:
                from stt_engine import STTEngine
            except ImportError:
                # If running from Pipeline folder
                sys.path.insert(0, str(PROJECT_ROOT / "STT"))
                from STT.stt_engine import STTEngine

        print(f"🎤 Initializing STT module ({backend}/{model_size})")

        self.engine = STTEngine(
            backend=backend,
            whisper_model=model_size,
            language=language,
            device=device
        )
        self.language = language

    def transcribe(self, audio: np.ndarray, sr: int = 16000) -> STTResult:
        """
        Transcribe audio array

        Args:
            audio: Audio samples
            sr: Sample rate

        Returns:
            STTResult with transcription
        """
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
        # LLM settings
        llm_backend: str = None,
        llm_model: str = None,
        llm_api_key: str = None,
        enable_llm: bool = True
    ):
        """
        Initialize EVA Pipeline

        Args:
            ser_checkpoint: Path to trained SER model
            stt_backend: STT backend ("whisper" or "vosk")
            stt_model: Whisper model size
            language: Language code
            device: Device for computation
            emotion_threshold: Threshold for dominant emotions
            parallel: Run STT and SER in parallel
            prompts_dir: Directory containing prompt files
            llm_backend: LLM backend (groq, gemini, ollama, etc.) - auto if None
            llm_model: LLM model name
            llm_api_key: API key for LLM (or use .env)
            enable_llm: Whether to enable LLM response generation
        """
        self.emotion_threshold = emotion_threshold
        self.parallel = parallel
        self.language = language
        self.enable_llm = enable_llm

        print("\n" + "=" * 60)
        print("🚀 Initializing EVA Pipeline")
        print("=" * 60)

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

        print("=" * 60)
        print("✅ EVA Pipeline ready!")
        print("=" * 60 + "\n")

    def _init_llm(self, backend: str, model: str, api_key: str):
        """Initialize LLM engine"""
        try:
            # Try different import paths
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

            if not self.llm.is_available():
                print("⚠️  LLM not available - responses will not be generated")
                self.llm = None

        except Exception as e:
            print(f"⚠️  Failed to initialize LLM: {e}")
            self.llm = None

    def _load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file"""
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)
        return audio, sr

    def _process_parallel(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Tuple[STTResult, SERResult]:
        """Process STT and SER in parallel using threads"""

        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            stt_future = executor.submit(self.stt.transcribe, audio, sr)
            ser_future = executor.submit(self.ser.predict, audio, sr, self.emotion_threshold)

            # Wait for results
            stt_result = stt_future.result()
            ser_result = ser_future.result()

        return stt_result, ser_result

    def _process_sequential(
        self,
        audio: np.ndarray,
        sr: int
    ) -> Tuple[STTResult, SERResult]:
        """Process STT and SER sequentially"""
        stt_result = self.stt.transcribe(audio, sr)
        ser_result = self.ser.predict(audio, sr, self.emotion_threshold)
        return stt_result, ser_result

    def _generate_llm_prompt(
        self,
        text: str,
        emotions: Dict[str, float],
        dominant_emotions: Dict[str, float]
    ) -> str:
        """
        Generate context-aware prompt for LLM

        Args:
            text: Transcribed text
            emotions: All emotion probabilities
            dominant_emotions: Emotions above threshold

        Returns:
            Formatted prompt for LLM
        """
        # Get top 3 emotions
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        top_emotions = sorted_emotions[:3]

        # Format emotion description
        emotion_desc = ", ".join([
            f"{emotion} ({prob * 100:.0f}%)"
            for emotion, prob in top_emotions
        ])

        # Determine emotional context
        primary_emotion = top_emotions[0][0] if top_emotions else "Neutral"
        primary_intensity = top_emotions[0][1] if top_emotions else 0.0

        # Get guidelines from prompt manager
        style_guide = self.prompt_manager.get_guidelines_for_emotion(
            primary_emotion,
            primary_intensity
        )

        # Build prompt
        prompt = f"""System Context: {self.prompt_manager.system_context}

═══════════════════════════════════════════════════════════════════════
EMOTIONAL ANALYSIS FROM VOICE
═══════════════════════════════════════════════════════════════════════
Detected emotions: {emotion_desc}
Primary emotion: {primary_emotion} ({primary_intensity * 100:.0f}% confidence)

═══════════════════════════════════════════════════════════════════════
RESPONSE GUIDELINES
═══════════════════════════════════════════════════════════════════════
{style_guide}

General principles:
{self.prompt_manager.general_principles}

═══════════════════════════════════════════════════════════════════════
USER'S MESSAGE
═══════════════════════════════════════════════════════════════════════
{text}

═══════════════════════════════════════════════════════════════════════
YOUR EMPATHIC RESPONSE:
"""

        return prompt

    def process(self, audio_path: str, generate_response: bool = True) -> PipelineResult:
        """
        Process audio file through the full pipeline

        Args:
            audio_path: Path to audio file
            generate_response: Whether to generate LLM response

        Returns:
            PipelineResult with all outputs
        """
        start_time = time.time()

        print(f"🎧 Processing: {Path(audio_path).name}")

        # Load audio
        audio, sr = self._load_audio(audio_path)
        print(f"   Duration: {len(audio) / sr:.2f}s")

        # Process STT and SER
        if self.parallel:
            print("   Running STT + SER in parallel...")
            stt_result, ser_result = self._process_parallel(audio, sr)
        else:
            print("   Running STT + SER sequentially...")
            stt_result, ser_result = self._process_sequential(audio, sr)

        # Generate LLM prompt
        llm_prompt = self._generate_llm_prompt(
            stt_result.text,
            ser_result.emotions,
            ser_result.dominant_emotions
        )

        # Generate LLM response
        llm_response = None
        llm_result = None

        if generate_response and self.llm and self.llm.is_available():
            print("   Generating empathic response...")
            llm_result = self._generate_response(llm_prompt)
            if llm_result:
                llm_response = llm_result.response

        total_time = time.time() - start_time

        # Build result
        result = PipelineResult(
            text=stt_result.text,
            emotions=ser_result.emotions,
            dominant_emotions=ser_result.dominant_emotions,
            llm_prompt=llm_prompt,
            llm_response=llm_response,
            stt_confidence=stt_result.confidence,
            total_processing_time=total_time,
            stt_result=stt_result,
            ser_result=ser_result,
            llm_result=llm_result
        )

        # Print summary
        self._print_summary(result)

        return result

    def _generate_response(self, prompt: str) -> Optional[LLMResult]:
        """Generate LLM response from prompt"""
        if not self.llm:
            return None

        try:
            response = self.llm.generate(
                prompt,
                max_tokens=256,
                temperature=0.7
            )

            return LLMResult(
                response=response.text,
                model=response.model,
                latency=response.latency,
                tokens_used=response.tokens_used
            )
        except Exception as e:
            print(f"   ⚠️  LLM error: {e}")
            return None

    def process_array(
        self,
        audio: np.ndarray,
        sr: int = 16000,
        generate_response: bool = True
    ) -> PipelineResult:
        """
        Process audio array through the pipeline

        Args:
            audio: Audio samples (mono, float)
            sr: Sample rate
            generate_response: Whether to generate LLM response

        Returns:
            PipelineResult
        """
        start_time = time.time()

        # Process
        if self.parallel:
            stt_result, ser_result = self._process_parallel(audio, sr)
        else:
            stt_result, ser_result = self._process_sequential(audio, sr)

        # Generate prompt
        llm_prompt = self._generate_llm_prompt(
            stt_result.text,
            ser_result.emotions,
            ser_result.dominant_emotions
        )

        # Generate LLM response
        llm_response = None
        llm_result = None

        if generate_response and self.llm and self.llm.is_available():
            llm_result = self._generate_response(llm_prompt)
            if llm_result:
                llm_response = llm_result.response

        total_time = time.time() - start_time

        return PipelineResult(
            text=stt_result.text,
            emotions=ser_result.emotions,
            dominant_emotions=ser_result.dominant_emotions,
            llm_prompt=llm_prompt,
            llm_response=llm_response,
            stt_confidence=stt_result.confidence,
            total_processing_time=total_time,
            stt_result=stt_result,
            ser_result=ser_result,
            llm_result=llm_result
        )

    def _print_summary(self, result: PipelineResult):
        """Print processing summary"""
        print("\n" + "─" * 60)
        print("📊 PROCESSING SUMMARY")
        print("─" * 60)

        # Text
        print(f"\n📝 Transcription:")
        print(f"   \"{result.text}\"")
        if result.stt_confidence:
            print(f"   Confidence: {result.stt_confidence:.1%}")

        # Emotions
        print(f"\n🎭 Emotions:")
        sorted_emotions = sorted(
            result.emotions.items(),
            key=lambda x: x[1],
            reverse=True
        )
        for emotion, prob in sorted_emotions[:4]:
            bar = "█" * int(prob * 20)
            print(f"   {emotion:<12} [{bar:<20}] {prob * 100:5.1f}%")

        # LLM Response
        if result.llm_response:
            print(f"\n💬 EVA's Response:")
            # Word wrap the response
            response_lines = result.llm_response.strip().split('\n')
            for line in response_lines:
                print(f"   {line}")

        # Timing
        print(f"\n⏱️  Timing:")
        print(f"   STT: {result.stt_result.processing_time:.2f}s")
        print(f"   SER: {result.ser_result.processing_time:.2f}s")
        if result.llm_result:
            print(f"   LLM: {result.llm_result.latency:.2f}s ({result.llm_result.model})")
        print(f"   Total: {result.total_processing_time:.2f}s")

        print("─" * 60 + "\n")


# --------------------------
# Quick Test
# --------------------------
def test_pipeline():
    """Quick test of the pipeline"""
    print("\n" + "=" * 60)
    print("🧪 EVA Pipeline Test")
    print("=" * 60)

    # Check for required files (relative to project root)
    checkpoint = PROJECT_ROOT / "checkpoints" / "best_model.pth"
    test_audio = PROJECT_ROOT / "test_audio.wav"

    if not checkpoint.exists():
        print(f"❌ Checkpoint not found: {checkpoint}")
        print("   Train a model first or update the path")
        return

    if not test_audio.exists():
        print(f"❌ Test audio not found: {test_audio}")
        return

    # Initialize pipeline
    pipeline = EVAPipeline(
        ser_checkpoint=str(checkpoint),
        stt_model="base",
        language="vi",
        parallel=True,
        enable_llm=True  # Enable LLM
    )

    # Process
    result = pipeline.process(str(test_audio))

    # Show results
    if result.llm_response:
        print("\n" + "=" * 60)
        print("🤖 EVA'S EMPATHIC RESPONSE")
        print("=" * 60)
        print(result.llm_response)
    else:
        print("\n" + "=" * 60)
        print("📋 GENERATED LLM PROMPT (no LLM available)")
        print("=" * 60)
        print(result.llm_prompt)

    print("\n✅ Pipeline test complete!")


# --------------------------
# CLI Interface
# --------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EVA Pipeline - STT + SER + LLM")
    parser.add_argument("audio", nargs="?", default=None, help="Audio file to process")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth", help="SER model checkpoint")
    parser.add_argument("--stt-model", default="base", help="Whisper model size")
    parser.add_argument("--language", default="vi", help="Language code")
    parser.add_argument("--sequential", action="store_true", help="Run sequentially instead of parallel")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM response generation")
    parser.add_argument("--llm-backend", default=None, help="LLM backend (groq, gemini, ollama)")
    parser.add_argument("--llm-model", default=None, help="LLM model name")
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
            llm_model=args.llm_model
        )
        result = pipeline.process(args.audio)

        if result.llm_response:
            print("\n🤖 EVA's Response:")
            print(result.llm_response)
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
        print("\nSetup LLM:")
        print("  1. Get free API key from https://console.groq.com/keys")
        print("  2. Create .env file: GROQ_API_KEY=your_key")
        print("  3. Run: python eva_pipeline.py --test")
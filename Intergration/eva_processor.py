"""
EVA Integration Module

Combines Speech-to-Text (STT) + Speech Emotion Recognition (SER)
for end-to-end audio analysis
"""

import torch
import numpy as np
import librosa
from pathlib import Path
from typing import Dict, Tuple, Optional
import sys

# Add parent directory to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from console import console, Colors
except ImportError:
    # Fallback console
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
        def header(self, title, width=70):
            print(f"\n{'='*width}")
            print(f"{title}")
            print(f"{'='*width}")
        def subheader(self, title, width=70):
            print(f"\n{'-'*width}")
            print(f"{title}")
            print(f"{'-'*width}")
        def item(self, label, value, indent=1): print(f"{'  '*indent}{label}: {value}")
        def emotion(self, name, score, width=20):
            bar = "█" * int(width * score) + "░" * (width - int(width * score))
            print(f"      {name:12s} [{bar}] {score*100:5.1f}%")
    console = Console()

from STT.stt_engine import STTEngine
from VAE.inference import EmotionRecognizer


class EVAProcessor:
    """
    End-to-End EVA Processor

    Pipeline:
    1. Audio Input
    2. STT: Extract text content
    3. SER: Analyze emotional state
    4. Generate context-aware prompt for LLM
    """

    def __init__(
        self,
        # STT config
        stt_backend: str = "whisper",
        stt_model: str = "base",
        stt_language: str = "vi",
        # SER config
        ser_checkpoint: str = "checkpoints/best_model.pth",
        ser_device: str = "auto",
        # Optional Vosk path
        vosk_model_path: Optional[str] = None
    ):
        """
        Initialize EVA Processor

        Args:
            stt_backend: "whisper" or "vosk"
            stt_model: Whisper model size
            stt_language: Language code
            ser_checkpoint: Path to SER model checkpoint
            ser_device: Device for SER model
            vosk_model_path: Path to Vosk model (if using Vosk)
        """
        console.header("Initializing EVA Processor")

        # Initialize STT
        console.info("[1/2] Setting up Speech-to-Text...")
        self.stt = STTEngine(
            backend=stt_backend,
            whisper_model=stt_model,
            vosk_model_path=vosk_model_path,
            device=ser_device,
            language=stt_language
        )

        # Initialize SER
        console.info("[2/2] Setting up Speech Emotion Recognition...")
        self.ser = EmotionRecognizer(
            checkpoint_path=ser_checkpoint,
            device=ser_device
        )

        console.success("EVA Processor initialized successfully!")
        print("=" * 70 + "\n")

    def process_audio_file(self, audio_path: str) -> Dict:
        """
        Process audio file through full pipeline

        Args:
            audio_path: Path to audio file

        Returns:
            Dictionary with transcription, emotions, and LLM prompt
        """
        console.info(f"Processing: {audio_path}")

        # Load audio
        audio, sr = librosa.load(audio_path, sr=16000, mono=True)

        # Run STT
        console.info("Running Speech-to-Text...", indent=1)
        transcription = self.stt.transcribe(audio_path)

        # Run SER
        console.info("Running Emotion Recognition...", indent=1)
        emotions_dict, latent = self.ser.predict(audio)

        # Get dominant emotions (>50%)
        dominant_emotions = {
            k: v for k, v in emotions_dict.items()
            if v > 0.5
        }

        # Build LLM prompt
        llm_prompt = self._build_llm_prompt(
            transcription['text'],
            emotions_dict
        )

        # Package results
        results = {
            'transcription': transcription,
            'emotions': {
                'all_emotions': emotions_dict,
                'dominant_emotions': dominant_emotions,
                'latent_vector': latent
            },
            'llm_prompt': llm_prompt
        }

        return results

    def process_audio_array(
        self,
        audio: np.ndarray,
        sr: int = 16000
    ) -> Dict:
        """
        Process audio array through full pipeline

        Args:
            audio: Audio samples as numpy array
            sr: Sample rate

        Returns:
            Dictionary with transcription, emotions, and LLM prompt
        """
        # Resample if needed
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
            sr = 16000

        # Run STT
        transcription = self.stt.transcribe_array(audio, sr=sr)

        # Run SER
        emotions_dict, latent = self.ser.predict(audio)

        # Get dominant emotions
        dominant_emotions = {
            k: v for k, v in emotions_dict.items()
            if v > 0.5
        }

        # Build LLM prompt
        llm_prompt = self._build_llm_prompt(
            transcription['text'],
            emotions_dict
        )

        # Package results
        results = {
            'transcription': transcription,
            'emotions': {
                'all_emotions': emotions_dict,
                'dominant_emotions': dominant_emotions,
                'latent_vector': latent
            },
            'llm_prompt': llm_prompt
        }

        return results

    def _build_llm_prompt(
        self,
        text: str,
        emotions: Dict[str, float]
    ) -> str:
        """
        Build a context-aware prompt for LLM

        Args:
            text: Transcribed text
            emotions: Emotion probabilities dict

        Returns:
            Formatted prompt string
        """
        # Sort emotions by probability
        sorted_emotions = sorted(
            emotions.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # Get top emotion
        top_emotion = sorted_emotions[0][0]
        top_prob = sorted_emotions[0][1]

        # Format emotion context
        emotion_context = ", ".join([
            f"{e}: {p*100:.1f}%"
            for e, p in sorted_emotions[:3]
        ])

        # Build prompt
        prompt = f"""You are EVA, an empathic voice assistant designed for emotional support.

CONTEXT:
- User's detected emotional state: {top_emotion} ({top_prob*100:.1f}% confidence)
- Emotion breakdown: {emotion_context}
- User's message: "{text}"

GUIDELINES:
- Respond with empathy and understanding
- Acknowledge the user's emotional state naturally
- If the user seems distressed, offer supportive words
- Keep responses conversational and warm
- Do not diagnose or provide medical advice
- If crisis indicators detected, suggest professional support

YOUR RESPONSE:"""

        return prompt

    def format_report(self, results: Dict) -> str:
        """
        Format processing results as a readable report

        Args:
            results: Output from process_audio_*()

        Returns:
            Formatted string report
        """
        report = []
        report.append("\n" + "=" * 70)
        report.append("EVA ANALYSIS REPORT")
        report.append("=" * 70)

        # Transcription
        report.append("\n[TRANSCRIPTION]")
        report.append(f"   Text: {results['transcription']['text']}")
        report.append(f"   Language: {results['transcription']['language']}")

        if results['transcription']['confidence']:
            conf = results['transcription']['confidence']
            report.append(f"   Confidence: {conf:.2%}")

        # Emotions
        report.append("\n[EMOTION ANALYSIS]")
        emotions = results['emotions']['all_emotions']
        sorted_emotions = sorted(
            emotions.items(),
            key=lambda x: x[1],
            reverse=True
        )

        report.append("   All emotions:")
        for emotion, prob in sorted_emotions:
            bar = "█" * int(prob * 20)
            report.append(f"      {emotion:12s} [{bar:20s}] {prob * 100:5.1f}%")

        dominant = results['emotions']['dominant_emotions']
        if dominant:
            report.append("\n   Dominant emotions (>50%):")
            for emotion, prob in dominant.items():
                report.append(f"      - {emotion}: {prob * 100:.1f}%")

        # LLM Prompt preview
        report.append("\n[LLM PROMPT] (preview)")
        prompt_lines = results['llm_prompt'].split('\n')[:10]
        for line in prompt_lines:
            report.append(f"   {line}")

        if len(results['llm_prompt'].split('\n')) > 10:
            report.append("   ...")

        report.append("\n" + "=" * 70)

        return "\n".join(report)


# --------------------------
# Test Function
# --------------------------
def test_processor():
    """Test the EVA processor"""
    console.header("EVA Processor Test")

    # Check for test file
    test_audio = PROJECT_ROOT / "test_audio.wav"
    checkpoint = PROJECT_ROOT / "checkpoints" / "best_model.pth"

    if not checkpoint.exists():
        console.error(f"Checkpoint not found: {checkpoint}")
        return

    if not test_audio.exists():
        console.error(f"Test audio not found: {test_audio}")
        console.info("Please provide a test audio file")
        return

    # Initialize
    processor = EVAProcessor(
        stt_backend="whisper",
        stt_model="base",
        stt_language="vi",
        ser_checkpoint=str(checkpoint)
    )

    # Process
    results = processor.process_audio_file(str(test_audio))

    # Print report
    report = processor.format_report(results)
    print(report)

    console.success("Test complete!")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "--test":
            test_processor()
        else:
            # Process provided audio file
            audio_path = sys.argv[1]

            processor = EVAProcessor(
                stt_backend="whisper",
                stt_model="base",
                stt_language="vi"
            )

            results = processor.process_audio_file(audio_path)
            report = processor.format_report(results)
            print(report)
    else:
        print("EVA Processor - Audio Analysis Pipeline")
        print("=" * 50)
        print("\nUsage:")
        print("  python eva_processor.py audio.wav    # Process audio file")
        print("  python eva_processor.py --test       # Run test")
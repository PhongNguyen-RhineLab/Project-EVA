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
sys.path.append(str(Path(__file__).parent.parent))

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
        print("\n" + "=" * 70)
        print("🤖 Initializing EVA Processor")
        print("=" * 70)

        # Initialize STT
        print("\n1️⃣  Setting up Speech-to-Text...")
        self.stt = STTEngine(
            backend=stt_backend,
            whisper_model=stt_model,
            vosk_model_path=vosk_model_path,
            device=ser_device,
            language=stt_language
        )

        # Initialize SER
        print("\n2️⃣  Setting up Speech Emotion Recognition...")
        self.ser = EmotionRecognizer(
            checkpoint_path=ser_checkpoint,
            device=ser_device
        )

        print("\n✅ EVA Processor initialized successfully!")
        print("=" * 70)

    def process_audio_file(
            self,
            audio_path: str,
            emotion_threshold: float = 0.3
    ) -> Dict:
        """
        Process audio file through full pipeline

        Args:
            audio_path: Path to audio file
            emotion_threshold: Threshold for emotion detection

        Returns:
            dict with:
                - transcription: STT results (text, language, confidence)
                - emotions: SER results (emotion dict, dominant, latent)
                - llm_prompt: Context-aware prompt for LLM
        """
        audio_path = Path(audio_path)
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        print(f"\n🎙️  Processing: {audio_path.name}")
        print("-" * 70)

        # Step 1: Speech-to-Text
        print("📝 Step 1: Transcribing audio...")
        transcription = self.stt.transcribe(str(audio_path))

        print(f"   Text: {transcription['text']}")
        print(f"   Language: {transcription['language']}")
        if transcription['confidence']:
            print(f"   Confidence: {transcription['confidence']:.2%}")

        # Step 2: Speech Emotion Recognition
        print("\n😊 Step 2: Analyzing emotions...")
        emotions_dict, dominant_emotions, latent = self.ser.predict(
            str(audio_path),
            threshold=emotion_threshold
        )

        # Display top emotions
        sorted_emotions = sorted(
            emotions_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        for emotion, prob in sorted_emotions:
            bar = "█" * int(prob * 20)
            print(f"   {emotion:12s} [{bar:20s}] {prob * 100:5.1f}%")

        # Step 3: Generate LLM prompt
        print("\n🤖 Step 3: Generating LLM prompt...")
        llm_prompt = self.ser.generate_llm_prompt(
            transcription['text'],
            emotions_dict
        )

        print("   ✅ Prompt generated")

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
            audio_array: np.ndarray,
            sr: int = 16000,
            emotion_threshold: float = 0.3
    ) -> Dict:
        """
        Process audio from numpy array

        Args:
            audio_array: Audio samples
            sr: Sample rate
            emotion_threshold: Threshold for emotion detection

        Returns:
            Same as process_audio_file()
        """
        print(f"\n🎙️  Processing audio array (sr={sr}Hz)")
        print("-" * 70)

        # Step 1: STT
        print("📝 Step 1: Transcribing audio...")
        transcription = self.stt.transcribe_array(audio_array, sr=sr)

        print(f"   Text: {transcription['text']}")

        # Step 2: SER
        # Need to save array temporarily for SER
        import tempfile
        import soundfile as sf

        print("\n😊 Step 2: Analyzing emotions...")

        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
            sf.write(tmp.name, audio_array, sr)
            tmp_path = tmp.name

        try:
            emotions_dict, dominant_emotions, latent = self.ser.predict(
                tmp_path,
                threshold=emotion_threshold
            )
        finally:
            Path(tmp_path).unlink()  # Clean up

        # Display emotions
        sorted_emotions = sorted(
            emotions_dict.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]

        for emotion, prob in sorted_emotions:
            bar = "█" * int(prob * 20)
            print(f"   {emotion:12s} [{bar:20s}] {prob * 100:5.1f}%")

        # Step 3: Generate prompt
        print("\n🤖 Step 3: Generating LLM prompt...")
        llm_prompt = self.ser.generate_llm_prompt(
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
        report.append("📊 EVA ANALYSIS REPORT")
        report.append("=" * 70)

        # Transcription
        report.append("\n📝 TRANSCRIPTION:")
        report.append(f"   Text: {results['transcription']['text']}")
        report.append(f"   Language: {results['transcription']['language']}")

        if results['transcription']['confidence']:
            conf = results['transcription']['confidence']
            report.append(f"   Confidence: {conf:.2%}")

        # Emotions
        report.append("\n😊 EMOTION ANALYSIS:")
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
        report.append("\n🤖 LLM PROMPT (preview):")
        prompt_lines = results['llm_prompt'].split('\n')[:10]
        for line in prompt_lines:
            report.append(f"   {line}")

        if len(results['llm_prompt'].split('\n')) > 10:
            report.append("   ...")

        report.append("\n" + "=" * 70)

        return "\n".join(report)


# --------------------------
# Example Usage
# --------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="EVA Audio Processor")
    parser.add_argument(
        "--audio",
        type=str,
        help="Path to audio file",
        default=None
    )
    parser.add_argument(
        "--stt-backend",
        type=str,
        choices=["whisper", "vosk"],
        default="whisper",
        help="STT backend"
    )
    parser.add_argument(
        "--stt-model",
        type=str,
        default="base",
        help="Whisper model size"
    )
    parser.add_argument(
        "--ser-checkpoint",
        type=str,
        default="checkpoints/best_model.pth",
        help="Path to SER checkpoint"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="vi",
        help="Language code"
    )

    args = parser.parse_args()

    # Initialize processor
    eva = EVAProcessor(
        stt_backend=args.stt_backend,
        stt_model=args.stt_model,
        stt_language=args.language,
        ser_checkpoint=args.ser_checkpoint
    )

    # Process audio
    if args.audio:
        results = eva.process_audio_file(args.audio)
        print(eva.format_report(results))
    else:
        print("\n⚠️  No audio file provided")
        print("Usage: python integration.py --audio path/to/audio.wav")

        # Demo with dummy audio
        print("\n🎭 Running demo with synthetic audio...")

        # Create dummy audio
        duration = 3
        sr = 16000
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.3 * np.sin(2 * np.pi * 440 * t)  # A4 note

        try:
            results = eva.process_audio_array(audio, sr=sr)
            print(eva.format_report(results))
        except Exception as e:
            print(f"\n❌ Demo failed: {e}")
            print("This is expected if SER checkpoint doesn't exist yet")
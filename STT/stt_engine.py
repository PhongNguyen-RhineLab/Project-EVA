"""
Speech-to-Text Engine for EVA Project

Supports:
1. OpenAI Whisper (primary)
2. Vosk (fallback for offline)
"""

import torch
import numpy as np
import librosa
from pathlib import Path
from typing import Optional, Dict, Tuple
import warnings

warnings.filterwarnings('ignore')


class WhisperSTT:
    """
    OpenAI Whisper-based Speech-to-Text
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        language: str = "vi"  # Vietnamese by default
    ):
        """
        Initialize Whisper STT

        Args:
            model_size: One of ["tiny", "base", "small", "medium", "large"]
            device: "cuda", "cpu", or "auto"
            language: Language code (e.g., "vi", "en")
        """
        self.model_size = model_size
        self.language = language

        # Set device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"🎤 Initializing Whisper STT ({model_size}) on {self.device}")

        # Load Whisper model
        try:
            import whisper
            self.model = whisper.load_model(model_size, device=self.device)
            self.whisper = whisper
            print(f"✅ Whisper model loaded successfully")
        except ImportError:
            raise ImportError(
                "Whisper not installed. Install with: pip install openai-whisper"
            )

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        task: str = "transcribe"
    ) -> Dict:
        """
        Transcribe audio file to text

        Args:
            audio_path: Path to audio file
            language: Override default language (optional)
            task: "transcribe" or "translate" (to English)

        Returns:
            dict with:
                - text: Transcribed text
                - language: Detected language
                - segments: List of segments with timestamps
                - confidence: Average confidence score
        """
        if language is None:
            language = self.language

        print(f"🎧 Transcribing: {Path(audio_path).name}")

        # Transcribe with Whisper
        # Note: verbose=False suppresses most output, but progress bar may still appear
        # This is normal Whisper behavior for processing audio chunks
        result = self.model.transcribe(
            audio_path,
            language=language,
            task=task,
            verbose=False
        )

        # Calculate average confidence from segments
        if 'segments' in result:
            confidences = []
            for seg in result['segments']:
                if 'no_speech_prob' in seg:
                    # Convert no_speech_prob to confidence
                    confidences.append(1.0 - seg['no_speech_prob'])

            avg_confidence = np.mean(confidences) if confidences else None
        else:
            avg_confidence = None

        output = {
            'text': result['text'].strip(),
            'language': result.get('language', language),
            'segments': result.get('segments', []),
            'confidence': avg_confidence
        }

        return output

    def transcribe_array(
        self,
        audio_array: np.ndarray,
        sr: int = 16000,
        language: Optional[str] = None
    ) -> Dict:
        """
        Transcribe audio from numpy array

        Args:
            audio_array: Audio samples (mono)
            sr: Sample rate
            language: Language code

        Returns:
            Same as transcribe()
        """
        if language is None:
            language = self.language

        # Ensure audio is float32 (Whisper requirement)
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)

        # Whisper expects 16kHz audio
        if sr != 16000:
            audio_array = librosa.resample(
                audio_array,
                orig_sr=sr,
                target_sr=16000
            )

        # Transcribe
        result = self.model.transcribe(
            audio_array,
            language=language,
            verbose=False
        )

        # Calculate confidence
        if 'segments' in result:
            confidences = [
                1.0 - seg.get('no_speech_prob', 0.0)
                for seg in result['segments']
            ]
            avg_confidence = np.mean(confidences) if confidences else None
        else:
            avg_confidence = None

        return {
            'text': result['text'].strip(),
            'language': result.get('language', language),
            'segments': result.get('segments', []),
            'confidence': avg_confidence
        }


class VoskSTT:
    """
    Vosk-based Speech-to-Text (offline alternative)
    """

    def __init__(
        self,
        model_path: str,
        sample_rate: int = 16000
    ):
        """
        Initialize Vosk STT

        Args:
            model_path: Path to Vosk model directory
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate

        print(f"🎤 Initializing Vosk STT from {model_path}")

        try:
            from vosk import Model, KaldiRecognizer
            import json

            self.Model = Model
            self.KaldiRecognizer = KaldiRecognizer
            self.json = json

            # Load model
            if not Path(model_path).exists():
                raise FileNotFoundError(f"Vosk model not found: {model_path}")

            self.model = Model(model_path)
            print(f"✅ Vosk model loaded successfully")

        except ImportError:
            raise ImportError(
                "Vosk not installed. Install with: pip install vosk"
            )

    def transcribe(
        self,
        audio_path: str
    ) -> Dict:
        """
        Transcribe audio file using Vosk

        Args:
            audio_path: Path to audio file

        Returns:
            dict with:
                - text: Transcribed text
                - language: "unknown" (Vosk doesn't detect language)
                - confidence: Average confidence
        """
        # Load audio
        audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)

        # Convert to int16
        audio_int16 = (audio * 32767).astype(np.int16)

        # Create recognizer
        recognizer = self.KaldiRecognizer(self.model, self.sample_rate)
        recognizer.SetWords(True)  # Get word-level timestamps

        # Process audio in chunks
        chunk_size = 4000
        results = []

        for i in range(0, len(audio_int16), chunk_size):
            chunk = audio_int16[i:i + chunk_size].tobytes()

            if recognizer.AcceptWaveform(chunk):
                result = self.json.loads(recognizer.Result())
                if 'text' in result and result['text']:
                    results.append(result)

        # Final result
        final_result = self.json.loads(recognizer.FinalResult())
        if 'text' in final_result and final_result['text']:
            results.append(final_result)

        # Combine all text
        full_text = " ".join([r.get('text', '') for r in results])

        # Calculate average confidence
        confidences = []
        for r in results:
            if 'result' in r:  # Word-level results
                for word in r['result']:
                    if 'conf' in word:
                        confidences.append(word['conf'])

        avg_confidence = np.mean(confidences) if confidences else None

        return {
            'text': full_text.strip(),
            'language': 'unknown',
            'confidence': avg_confidence
        }

    def transcribe_array(
        self,
        audio_array: np.ndarray,
        sr: int = 16000
    ) -> Dict:
        """
        Transcribe from numpy array
        """
        # Resample if needed
        if sr != self.sample_rate:
            audio_array = librosa.resample(
                audio_array,
                orig_sr=sr,
                target_sr=self.sample_rate
            )

        # Convert to int16
        audio_int16 = (audio_array * 32767).astype(np.int16)

        # Create recognizer
        recognizer = self.KaldiRecognizer(self.model, self.sample_rate)
        recognizer.SetWords(True)

        # Process
        chunk_size = 4000
        results = []

        for i in range(0, len(audio_int16), chunk_size):
            chunk = audio_int16[i:i + chunk_size].tobytes()

            if recognizer.AcceptWaveform(chunk):
                result = self.json.loads(recognizer.Result())
                if 'text' in result and result['text']:
                    results.append(result)

        final_result = self.json.loads(recognizer.FinalResult())
        if 'text' in final_result and final_result['text']:
            results.append(final_result)

        full_text = " ".join([r.get('text', '') for r in results])

        # Calculate confidence
        confidences = []
        for r in results:
            if 'result' in r:
                for word in r['result']:
                    if 'conf' in word:
                        confidences.append(word['conf'])

        avg_confidence = np.mean(confidences) if confidences else None

        return {
            'text': full_text.strip(),
            'language': 'unknown',
            'confidence': avg_confidence
        }


class STTEngine:
    """
    Unified STT Engine with multiple backend support
    """

    def __init__(
        self,
        backend: str = "whisper",
        whisper_model: str = "base",
        vosk_model_path: Optional[str] = None,
        device: str = "auto",
        language: str = "vi"
    ):
        """
        Initialize STT Engine

        Args:
            backend: "whisper" or "vosk"
            whisper_model: Whisper model size
            vosk_model_path: Path to Vosk model (if using Vosk)
            device: Device for computation
            language: Language code
        """
        self.backend = backend
        self.language = language

        if backend == "whisper":
            self.engine = WhisperSTT(
                model_size=whisper_model,
                device=device,
                language=language
            )
        elif backend == "vosk":
            if vosk_model_path is None:
                raise ValueError("vosk_model_path required for Vosk backend")
            self.engine = VoskSTT(model_path=vosk_model_path)
        else:
            raise ValueError(f"Unknown backend: {backend}")

    def transcribe(self, audio_path: str, **kwargs) -> Dict:
        """Transcribe audio file"""
        return self.engine.transcribe(audio_path, **kwargs)

    def transcribe_array(self, audio_array: np.ndarray, sr: int = 16000, **kwargs) -> Dict:
        """Transcribe audio array"""
        return self.engine.transcribe_array(audio_array, sr, **kwargs)


# --------------------------
# Example Usage
# --------------------------
if __name__ == "__main__":
    import sys

    # Example 1: Whisper STT
    print("\n" + "=" * 60)
    print("Testing Whisper STT")
    print("=" * 60)

    stt = STTEngine(backend="whisper", whisper_model="base", language="vi")

    # Test with audio file (if exists)
    test_audio = "test_audio.wav"
    if Path(test_audio).exists():
        result = stt.transcribe(test_audio)

        print(f"\n📝 Transcription:")
        print(f"   Text: {result['text']}")
        print(f"   Language: {result['language']}")
        print(f"   Confidence: {result['confidence']:.2%}" if result['confidence'] else "   Confidence: N/A")

        if result['segments']:
            print(f"\n📊 Segments:")
            for i, seg in enumerate(result['segments'][:3]):  # Show first 3
                print(f"   [{seg['start']:.1f}s - {seg['end']:.1f}s]: {seg['text']}")
    else:
        print(f"⚠️  Test audio file not found: {test_audio}")
        print("   Create a test audio file to see the demo")

    print("\n" + "=" * 60)
    print("STT Engine ready!")
    print("=" * 60)
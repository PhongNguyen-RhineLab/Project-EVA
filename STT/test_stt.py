"""
Test script for STT Engine

Tests both Whisper and (optionally) Vosk backends
"""

import sys
import numpy as np
import librosa
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from STT.stt_engine import STTEngine


def create_dummy_audio(duration=3, sr=16000):
    """Create a dummy audio signal for testing"""
    t = np.linspace(0, duration, int(sr * duration))
    # Simple sine wave
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # A4 note
    return audio, sr


def test_whisper():
    """Test Whisper STT"""
    print("\n" + "=" * 70)
    print("🧪 Testing Whisper STT")
    print("=" * 70)

    # Initialize
    stt = STTEngine(
        backend="whisper",
        whisper_model="base",  # Options: tiny, base, small, medium, large
        language="vi",
        device="auto"
    )

    # Test 1: Transcribe from file
    print("\n📁 Test 1: Transcribe from file")
    print("-" * 70)

    test_file = "test_audio.wav"
    if Path(test_file).exists():
        result = stt.transcribe(test_file)
        display_result(result)
    else:
        print(f"⚠️  File not found: {test_file}")
        print("   Skipping file test...")

    # Test 2: Transcribe from numpy array
    print("\n🔢 Test 2: Transcribe from numpy array")
    print("-" * 70)

    dummy_audio, sr = create_dummy_audio()
    result = stt.transcribe_array(dummy_audio, sr=sr)
    print("✅ Successfully processed numpy array")
    display_result(result)

    # Test 3: Different languages
    print("\n🌍 Test 3: Multi-language support")
    print("-" * 70)

    languages = ["vi", "en"]
    for lang in languages:
        stt_lang = STTEngine(
            backend="whisper",
            whisper_model="base",
            language=lang,
            device="auto"
        )
        print(f"   Initialized for language: {lang} ✅")

    print("\n✅ Whisper tests completed!")


def test_vosk(model_path: str):
    """Test Vosk STT"""
    print("\n" + "=" * 70)
    print("🧪 Testing Vosk STT")
    print("=" * 70)

    if not Path(model_path).exists():
        print(f"⚠️  Vosk model not found: {model_path}")
        print("   Download from: https://alphacephei.com/vosk/models")
        return

    # Initialize
    stt = STTEngine(
        backend="vosk",
        vosk_model_path=model_path
    )

    # Test with file
    test_file = "test_audio.wav"
    if Path(test_file).exists():
        result = stt.transcribe(test_file)
        display_result(result)
    else:
        print(f"⚠️  File not found: {test_file}")

    print("\n✅ Vosk tests completed!")


def display_result(result: dict):
    """Pretty print STT result"""
    print(f"\n📝 Transcription Result:")
    print(f"   Text: {result['text']}")
    print(f"   Language: {result['language']}")

    if result['confidence'] is not None:
        print(f"   Confidence: {result['confidence']:.2%}")
    else:
        print(f"   Confidence: N/A")

    if 'segments' in result and result['segments']:
        print(f"\n📊 Segments ({len(result['segments'])} total):")
        for i, seg in enumerate(result['segments'][:5]):  # Show first 5
            start = seg.get('start', 0)
            end = seg.get('end', 0)
            text = seg.get('text', '').strip()
            print(f"   [{start:6.2f}s - {end:6.2f}s]: {text}")

        if len(result['segments']) > 5:
            print(f"   ... and {len(result['segments']) - 5} more segments")


def benchmark_models():
    """Benchmark different Whisper model sizes"""
    print("\n" + "=" * 70)
    print("⚡ Benchmarking Whisper Models")
    print("=" * 70)

    models = ["tiny", "base", "small"]

    # Create test audio
    audio, sr = create_dummy_audio(duration=5)

    import time

    for model in models:
        print(f"\n🔍 Testing {model.upper()} model...")

        try:
            stt = STTEngine(
                backend="whisper",
                whisper_model=model,
                language="vi"
            )

            start_time = time.time()
            result = stt.transcribe_array(audio, sr=sr)
            elapsed = time.time() - start_time

            print(f"   ✅ Completed in {elapsed:.2f}s")
            print(f"   📝 Text: {result['text'][:50]}...")

        except Exception as e:
            print(f"   ❌ Error: {e}")

    print("\n✅ Benchmark completed!")


def main():
    """Main test function"""
    print("\n" + "=" * 70)
    print("🎤 EVA STT Engine - Test Suite")
    print("=" * 70)

    # Test Whisper (primary)
    try:
        test_whisper()
    except Exception as e:
        print(f"\n❌ Whisper test failed: {e}")

    # Test Vosk (optional)
    vosk_model = "models/vosk-model-small-vi-0.4"  # Example Vietnamese model
    if Path(vosk_model).exists():
        try:
            test_vosk(vosk_model)
        except Exception as e:
            print(f"\n❌ Vosk test failed: {e}")
    else:
        print(f"\n⏭️  Skipping Vosk test (model not found)")

    # Benchmark (optional)
    print("\n" + "=" * 70)
    user_input = input("Run benchmark? (y/n): ")
    if user_input.lower() == 'y':
        benchmark_models()

    print("\n" + "=" * 70)
    print("✅ All tests completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
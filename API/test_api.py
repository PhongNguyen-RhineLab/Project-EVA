"""
Test script for EVA API

Usage:
    1. Start the server: python API/eva_api.py
    2. Run tests: python API/test_api.py
"""

import requests
import sys
from pathlib import Path

# API base URL
BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    print("\n🔍 Testing /health...")

    try:
        response = requests.get(f"{BASE_URL}/health")
        data = response.json()

        print(f"   Status: {data['status']}")
        print(f"   Components:")
        for component, available in data['components'].items():
            icon = "✅" if available else "❌"
            print(f"      {component}: {icon}")

        return data['status'] == 'healthy'
    except requests.ConnectionError:
        print("   ❌ Could not connect to server")
        print("   Make sure the server is running: python API/eva_api.py")
        return False


def test_config():
    """Test config endpoint"""
    print("\n🔍 Testing /config...")

    response = requests.get(f"{BASE_URL}/config")
    data = response.json()

    print(f"   STT Model: {data['stt_model']}")
    print(f"   Language: {data['language']}")
    print(f"   LLM Backend: {data['llm_backend'] or 'None'}")
    print(f"   LLM Available: {'✅' if data['llm_available'] else '❌'}")

    return True


def test_process(audio_path: str):
    """Test full processing endpoint"""
    print(f"\n🔍 Testing /process with {audio_path}...")

    if not Path(audio_path).exists():
        print(f"   ❌ Audio file not found: {audio_path}")
        return False

    with open(audio_path, 'rb') as f:
        files = {'file': (Path(audio_path).name, f, 'audio/wav')}
        response = requests.post(f"{BASE_URL}/process", files=files)

    data = response.json()

    if not data.get('success'):
        print(f"   ❌ Error: {data.get('error', 'Unknown error')}")
        return False

    print(f"\n   📝 Transcription: \"{data['transcription']}\"")
    print(f"   🎯 Confidence: {data['stt_confidence']:.1%}" if data['stt_confidence'] else "")

    print(f"\n   🎭 Emotions:")
    for emotion in data['emotions'][:4]:
        print(f"      {emotion['emotion']}: {emotion['percentage']}")

    print(f"\n   🏆 Primary: {data['primary_emotion']} ({data['primary_emotion_score'] * 100:.1f}%)")

    if data.get('eva_response'):
        print(f"\n   💬 EVA's Response:")
        for line in data['eva_response'].split('\n'):
            print(f"      {line}")

    print(f"\n   ⏱️  Timing:")
    times = data['processing_times']
    print(f"      STT: {times['stt']:.2f}s")
    print(f"      SER: {times['ser']:.2f}s")
    if times.get('llm'):
        print(f"      LLM: {times['llm']:.2f}s")
    print(f"      Total: {times['total']:.2f}s")

    return True


def test_transcribe(audio_path: str):
    """Test transcription-only endpoint"""
    print(f"\n🔍 Testing /transcribe with {audio_path}...")

    if not Path(audio_path).exists():
        print(f"   ❌ Audio file not found: {audio_path}")
        return False

    with open(audio_path, 'rb') as f:
        files = {'file': (Path(audio_path).name, f, 'audio/wav')}
        response = requests.post(f"{BASE_URL}/transcribe", files=files)

    data = response.json()

    if not data.get('success'):
        print(f"   ❌ Error: {data.get('error', 'Unknown error')}")
        return False

    print(f"   📝 Text: \"{data['transcription']}\"")
    print(f"   🌍 Language: {data['language']}")
    print(f"   ⏱️  Time: {data['processing_time']:.2f}s")

    return True


def test_emotions(audio_path: str):
    """Test emotions-only endpoint"""
    print(f"\n🔍 Testing /emotions with {audio_path}...")

    if not Path(audio_path).exists():
        print(f"   ❌ Audio file not found: {audio_path}")
        return False

    with open(audio_path, 'rb') as f:
        files = {'file': (Path(audio_path).name, f, 'audio/wav')}
        response = requests.post(f"{BASE_URL}/emotions", files=files)

    data = response.json()

    if not data.get('success'):
        print(f"   ❌ Error: {data.get('error', 'Unknown error')}")
        return False

    print(f"   🎭 Emotions:")
    for emotion in data['emotions'][:4]:
        print(f"      {emotion['emotion']}: {emotion['percentage']}")

    print(f"   🏆 Primary: {data['primary_emotion']}")
    print(f"   ⏱️  Time: {data['processing_time']:.2f}s")

    return True


def test_chat():
    """Test text chat endpoint"""
    print("\n🔍 Testing /chat...")

    response = requests.post(
        f"{BASE_URL}/chat",
        params={
            "text": "Tôi cảm thấy hơi buồn hôm nay",
            "emotion": "Sad",
            "emotion_score": 0.7
        }
    )

    data = response.json()

    if not data.get('success'):
        print(f"   ❌ Error: {data.get('error', 'Unknown error')}")
        return False

    print(f"   💬 Response: {data['response'][:200]}...")
    print(f"   🤖 Model: {data['model']}")
    print(f"   ⏱️  Time: {data['processing_time']:.2f}s")

    return True


def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 EVA API Test Suite")
    print("=" * 60)

    # Find test audio
    project_root = Path(__file__).parent.parent
    test_audio = project_root / "test_audio.wav"

    # Run tests
    results = []

    # Health check first
    if not test_health():
        print("\n❌ Server not available. Start it with:")
        print("   python API/eva_api.py")
        return

    results.append(("Health", True))
    results.append(("Config", test_config()))

    if test_audio.exists():
        results.append(("Transcribe", test_transcribe(str(test_audio))))
        results.append(("Emotions", test_emotions(str(test_audio))))
        results.append(("Process", test_process(str(test_audio))))
    else:
        print(f"\n⚠️  Test audio not found: {test_audio}")
        print("   Skipping audio tests...")

    results.append(("Chat", test_chat()))

    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Results")
    print("=" * 60)

    passed = 0
    for name, success in results:
        icon = "✅" if success else "❌"
        print(f"   {icon} {name}")
        if success:
            passed += 1

    print(f"\n   Total: {passed}/{len(results)} passed")
    print("=" * 60)


if __name__ == "__main__":
    main()
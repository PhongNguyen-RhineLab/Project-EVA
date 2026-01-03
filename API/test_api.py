"""
Test script for EVA API

Usage:
    1. Start the server: python API/eva_api.py
    2. Run tests: python API/test_api.py
"""

import requests
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from console import console, Colors

# API base URL
BASE_URL = "http://localhost:8000"


def test_health():
    """Test health endpoint"""
    console.info("Testing /health...")

    try:
        response = requests.get(f"{BASE_URL}/health")
        data = response.json()

        console.item("Status", data['status'])
        print("   Components:")
        for component, available in data['components'].items():
            status = f"{Colors.GREEN}yes{Colors.RESET}" if available else f"{Colors.RED}no{Colors.RESET}"
            console.item(component, status, indent=2)

        return data['status'] == 'healthy'
    except requests.ConnectionError:
        console.error("Could not connect to server")
        console.info("Make sure the server is running: python API/eva_api.py", indent=1)
        return False


def test_config():
    """Test config endpoint"""
    console.info("Testing /config...")

    response = requests.get(f"{BASE_URL}/config")
    data = response.json()

    console.item("STT Model", data['stt_model'])
    console.item("Language", data['language'])
    console.item("LLM Backend", data['llm_backend'] or 'None')

    llm_status = f"{Colors.GREEN}yes{Colors.RESET}" if data['llm_available'] else f"{Colors.RED}no{Colors.RESET}"
    console.item("LLM Available", llm_status)

    return True


def test_process(audio_path: str):
    """Test full processing endpoint"""
    console.info(f"Testing /process with {audio_path}...")

    if not Path(audio_path).exists():
        console.error(f"Audio file not found: {audio_path}")
        return False

    with open(audio_path, 'rb') as f:
        files = {'file': (Path(audio_path).name, f, 'audio/wav')}
        response = requests.post(f"{BASE_URL}/process", files=files)

    data = response.json()

    if not data.get('success'):
        console.error(f"Error: {data.get('error', 'Unknown error')}")
        return False

    console.success("Processing successful")
    console.item("Transcription", data['transcription'])
    console.item("Language", data['language'])
    console.item("Primary Emotion", f"{data['primary_emotion']} ({data['primary_emotion_score']*100:.1f}%)")

    print()
    console.info("All emotions:")
    for emotion in data['emotions'][:5]:
        console.emotion(emotion['emotion'], emotion['score'])

    if data.get('eva_response'):
        print()
        console.info("EVA Response:")
        # Wrap long response
        response_text = data['eva_response']
        for line in response_text.split('\n'):
            print(f"   {line}")

    print()
    times = data['processing_times']
    console.item("STT time", f"{times['stt']:.3f}s")
    console.item("SER time", f"{times['ser']:.3f}s")
    if times.get('llm'):
        console.item("LLM time", f"{times['llm']:.3f}s")
    console.item("Total time", f"{times['total']:.3f}s")

    return True


def test_transcribe(audio_path: str):
    """Test transcription-only endpoint"""
    console.info(f"Testing /transcribe with {audio_path}...")

    if not Path(audio_path).exists():
        console.error(f"Audio file not found: {audio_path}")
        return False

    with open(audio_path, 'rb') as f:
        files = {'file': (Path(audio_path).name, f, 'audio/wav')}
        response = requests.post(f"{BASE_URL}/transcribe", files=files)

    data = response.json()

    if not data.get('success'):
        console.error(f"Error: {data.get('error', 'Unknown error')}")
        return False

    console.success("Transcription successful")
    console.item("Text", data['transcription'])
    console.item("Language", data['language'])
    console.item("Time", f"{data['processing_time']:.3f}s")

    return True


def test_emotions(audio_path: str):
    """Test emotions-only endpoint"""
    console.info(f"Testing /emotions with {audio_path}...")

    if not Path(audio_path).exists():
        console.error(f"Audio file not found: {audio_path}")
        return False

    with open(audio_path, 'rb') as f:
        files = {'file': (Path(audio_path).name, f, 'audio/wav')}
        response = requests.post(f"{BASE_URL}/emotions", files=files)

    data = response.json()

    if not data.get('success'):
        console.error(f"Error: {data.get('error', 'Unknown error')}")
        return False

    console.success("Emotion analysis successful")
    console.item("Primary", f"{data['primary_emotion']} ({data['primary_emotion_score']*100:.1f}%)")
    console.item("Time", f"{data['processing_time']:.3f}s")

    print()
    for emotion in data['emotions'][:5]:
        console.emotion(emotion['emotion'], emotion['score'])

    return True


def test_chat(text: str = "I'm feeling a bit stressed today"):
    """Test chat endpoint"""
    console.info(f"Testing /chat...")
    console.item("Input", text)

    response = requests.post(
        f"{BASE_URL}/chat",
        params={"text": text, "emotion": "stressed", "emotion_score": 0.7}
    )

    data = response.json()

    if not data.get('success'):
        console.error(f"Error: {data.get('error', 'Unknown error')}")
        return False

    console.success("Chat successful")
    console.item("Model", data.get('model', 'unknown'))
    console.item("Time", f"{data['processing_time']:.3f}s")

    print()
    console.info("Response:")
    for line in data['response'].split('\n'):
        print(f"   {line}")

    return True


def main():
    """Run all tests"""
    import argparse

    global BASE_URL
    parser = argparse.ArgumentParser(description="Test EVA API")
    parser.add_argument("--audio", help="Audio file for testing")
    parser.add_argument("--url", default=BASE_URL, help="API base URL")
    parser.add_argument("--test", choices=['health', 'config', 'process', 'transcribe', 'emotions', 'chat', 'all'],
                        default='all', help="Which test to run")

    args = parser.parse_args()
    BASE_URL = args.url
    console.header("EVA API Test Suite")

    results = {}

    # Health check first
    if args.test in ['all', 'health']:
        print()
        results['health'] = test_health()
        if not results['health'] and args.test == 'all':
            console.error("Server not available, skipping other tests")
            return

    if args.test in ['all', 'config']:
        print()
        results['config'] = test_config()

    # Audio-dependent tests
    if args.audio:
        if args.test in ['all', 'process']:
            print()
            results['process'] = test_process(args.audio)

        if args.test in ['all', 'transcribe']:
            print()
            results['transcribe'] = test_transcribe(args.audio)

        if args.test in ['all', 'emotions']:
            print()
            results['emotions'] = test_emotions(args.audio)
    elif args.test in ['process', 'transcribe', 'emotions']:
        console.warning("Audio file required for this test. Use --audio path/to/file.wav")

    if args.test in ['all', 'chat']:
        print()
        results['chat'] = test_chat()

    # Summary
    console.subheader("Test Results")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for test, success in results.items():
        status = f"{Colors.GREEN}PASS{Colors.RESET}" if success else f"{Colors.RED}FAIL{Colors.RESET}"
        print(f"  {test:12s} [{status}]")

    print()
    color = Colors.GREEN if passed == total else Colors.YELLOW
    print(f"  {color}Total: {passed}/{total} passed{Colors.RESET}")


if __name__ == "__main__":
    main()
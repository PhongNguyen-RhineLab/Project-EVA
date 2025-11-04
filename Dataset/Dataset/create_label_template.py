import pandas as pd
import os
from pathlib import Path

# Configuration
AUDIO_DIR = "D:/Dataset Collection/vivos/test/waves/VIVOSDEV02"  # Change this
OUTPUT_CSV = "D:/Dataset Collection/vivos/test/waves/VIVOSDEV02/VIVOSDEV02.csv"
DATASET_NAME = "VIVOSDEV02"  # Change this

# Emotion mapping (one-hot encoding)
EMOTION_MAP = {
    'neutral': [1, 0, 0, 0, 0, 0, 0, 0],
    'calm': [0, 1, 0, 0, 0, 0, 0, 0],
    'happy': [0, 0, 1, 0, 0, 0, 0, 0],
    'sad': [0, 0, 0, 1, 0, 0, 0, 0],
    'angry': [0, 0, 0, 0, 1, 0, 0, 0],
    'fearful': [0, 0, 0, 0, 0, 1, 0, 0],
    'disgust': [0, 0, 0, 0, 0, 0, 1, 0],
    'surprised': [0, 0, 0, 0, 0, 0, 0, 1]
}

EMOTION_COLS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']


def create_template():
    """Create a CSV template from audio files"""

    # Find all .wav files
    audio_files = list(Path(AUDIO_DIR).rglob('*.wav'))

    if not audio_files:
        print(f"❌ No .wav files found in {AUDIO_DIR}")
        return

    print(f"✅ Found {len(audio_files)} audio files")

    data = []
    for audio_file in audio_files:
        # Default to 'neutral' - YOU NEED TO MANUALLY EDIT THIS
        emotion = 'neutral'

        row = {
            'filename': str(audio_file.absolute()),
            'emotion': emotion,
            'dataset': DATASET_NAME,
        }

        # Add one-hot encoding
        for col, val in zip(EMOTION_COLS, EMOTION_MAP[emotion]):
            row[col] = val

        data.append(row)

    # Create DataFrame
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n📄 Template saved to: {OUTPUT_CSV}")
    print(f"📝 Total rows: {len(df)}")
    print("\n⚠️  IMPORTANT: Edit the CSV and change 'emotion' column for each file!")
    print("   Then re-run the script to update the one-hot encodings.\n")


if __name__ == "__main__":
    create_template()
"""
Dataset preparation script - Use pre-labeled datasets
No manual labeling required!
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split

# --------------------------
# Emotion Mapping (8 emotions for EVA)
# --------------------------
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


# --------------------------
# RAVDESS Dataset Parser
# --------------------------
def parse_ravdess(ravdess_dir):
    """
    Parse RAVDESS dataset (already labeled in filename)

    Filename format: 03-01-06-01-02-01-12.wav
    Position 3 (index 2): Emotion
        01 = neutral, 02 = calm, 03 = happy, 04 = sad,
        05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
    """
    data = []

    ravdess_emotions = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }

    for actor_folder in Path(ravdess_dir).glob('Actor_*'):
        for audio_file in actor_folder.glob('*.wav'):
            # Parse filename
            parts = audio_file.stem.split('-')
            emotion_code = parts[2]
            emotion = ravdess_emotions.get(emotion_code)

            if emotion:
                data.append({
                    'filename': str(audio_file),
                    'emotion': emotion,
                    **{col: val for col, val in zip(EMOTION_COLS, EMOTION_MAP[emotion])}
                })

    return pd.DataFrame(data)


# --------------------------
# TESS Dataset Parser
# --------------------------
def parse_tess(tess_dir):
    """
    Parse TESS dataset (already labeled in filename)

    Filename format: OAF_back_angry.wav
    Last part before .wav is the emotion
    """
    data = []

    tess_emotion_map = {
        'angry': 'angry',
        'disgust': 'disgust',
        'fear': 'fearful',
        'happy': 'happy',
        'neutral': 'neutral',
        'ps': 'surprised',  # pleasant surprise
        'sad': 'sad'
    }

    for audio_file in Path(tess_dir).rglob('*.wav'):
        # Parse filename
        emotion_raw = audio_file.stem.split('_')[-1].lower()
        emotion = tess_emotion_map.get(emotion_raw)

        if emotion:
            data.append({
                'filename': str(audio_file),
                'emotion': emotion,
                **{col: val for col, val in zip(EMOTION_COLS, EMOTION_MAP[emotion])}
            })

    return pd.DataFrame(data)


# --------------------------
# CREMA-D Dataset Parser
# --------------------------
def parse_crema(crema_dir):
    """
    Parse CREMA-D dataset (already labeled in filename)

    Filename format: 1001_DFA_ANG_XX.wav
    Third part is emotion code
    """
    data = []

    crema_emotion_map = {
        'ANG': 'angry',
        'DIS': 'disgust',
        'FEA': 'fearful',
        'HAP': 'happy',
        'NEU': 'neutral',
        'SAD': 'sad'
    }

    for audio_file in Path(crema_dir).glob('*.wav'):
        parts = audio_file.stem.split('_')
        if len(parts) >= 3:
            emotion_code = parts[2]
            emotion = crema_emotion_map.get(emotion_code)

            if emotion:
                data.append({
                    'filename': str(audio_file),
                    'emotion': emotion,
                    **{col: val for col, val in zip(EMOTION_COLS, EMOTION_MAP[emotion])}
                })

    return pd.DataFrame(data)


# --------------------------
# Combine All Datasets
# --------------------------
def prepare_combined_dataset(ravdess_dir=None, tess_dir=None, crema_dir=None,
                             output_dir='EVA_Dataset'):
    """
    Combine multiple pre-labeled datasets
    No manual labeling required!
    """
    all_data = []

    # Parse each dataset
    if ravdess_dir and os.path.exists(ravdess_dir):
        print(f"📁 Loading RAVDESS from {ravdess_dir}...")
        ravdess_df = parse_ravdess(ravdess_dir)
        print(f"   Found {len(ravdess_df)} samples")
        all_data.append(ravdess_df)

    if tess_dir and os.path.exists(tess_dir):
        print(f"📁 Loading TESS from {tess_dir}...")
        tess_df = parse_tess(tess_dir)
        print(f"   Found {len(tess_df)} samples")
        all_data.append(tess_df)

    if crema_dir and os.path.exists(crema_dir):
        print(f"📁 Loading CREMA-D from {crema_dir}...")
        crema_df = parse_crema(crema_dir)
        print(f"   Found {len(crema_df)} samples")
        all_data.append(crema_df)

    if not all_data:
        raise ValueError("No datasets found! Please download at least one dataset.")

    # Combine
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"\n✅ Total samples: {len(combined_df)}")

    # Show emotion distribution
    print("\n📊 Emotion Distribution:")
    emotion_counts = combined_df['emotion'].value_counts()
    for emotion, count in emotion_counts.items():
        print(f"   {emotion:12s}: {count:5d} ({count / len(combined_df) * 100:.1f}%)")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/processed_audio", exist_ok=True)
    os.makedirs(f"{output_dir}/labels", exist_ok=True)

    # Copy audio files to processed_audio
    print("\n📋 Copying audio files...")
    for idx, row in combined_df.iterrows():
        src = row['filename']
        dst = f"{output_dir}/processed_audio/{Path(src).name}"

        # Avoid duplicates by adding index if needed
        if os.path.exists(dst):
            dst = f"{output_dir}/processed_audio/{idx}_{Path(src).name}"

        shutil.copy2(src, dst)
        combined_df.at[idx, 'filename'] = Path(dst).name

    # Split train/val/test (70/15/15)
    train_df, temp_df = train_test_split(combined_df, test_size=0.3,
                                         stratify=combined_df['emotion'],
                                         random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5,
                                       stratify=temp_df['emotion'],
                                       random_state=42)

    # Save labels
    label_cols = ['filename'] + EMOTION_COLS
    train_df[label_cols].to_csv(f"{output_dir}/labels/train_labels.csv", index=False)
    val_df[label_cols].to_csv(f"{output_dir}/labels/val_labels.csv", index=False)
    test_df[label_cols].to_csv(f"{output_dir}/labels/test_labels.csv", index=False)

    print(f"\n✅ Dataset prepared successfully!")
    print(f"   Train: {len(train_df)} samples")
    print(f"   Val:   {len(val_df)} samples")
    print(f"   Test:  {len(test_df)} samples")
    print(f"\n📂 Output directory: {output_dir}/")

    return combined_df, train_df, val_df, test_df


# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    # Configure your dataset paths
    DATASETS = {
        'ravdess_dir': 'Dataset/prelabel_en/RAVDESS',
        'tess_dir': 'Dataset/prelabel_en/TESS',
        'crema_dir': 'Dataset/prelabel_en/CREMA-D',
    }

    # Prepare dataset (no manual labeling!)
    combined_df, train_df, val_df, test_df = prepare_combined_dataset(
        ravdess_dir=DATASETS.get('ravdess_dir'),
        tess_dir=DATASETS.get('tess_dir'),
        crema_dir=DATASETS.get('crema_dir'),
        output_dir='EVA_Dataset'
    )

    print("\n🎉 Ready to train! Run: python train.py")
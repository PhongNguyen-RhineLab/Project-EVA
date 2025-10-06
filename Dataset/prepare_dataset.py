"""
EVA Dataset Setup Script
Prepares RAVDESS, TESS, and CREMA-D for training
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
from collections import Counter

# --------------------------
# Configuration
# --------------------------
DATASETS = {
    'ravdess_dir': 'Dataset/prelabel_en/RAVDESS',
    'tess_dir': 'Dataset/prelabel_en/TESS',
    'crema_dir': 'Dataset/prelabel_en/CREMA-D',
}

OUTPUT_DIR = 'EVA_Dataset'

# 8 Emotions for EVA
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
# RAVDESS Parser
# --------------------------
def parse_ravdess(ravdess_dir):
    """
    RAVDESS filename format: 03-01-06-01-02-01-12.wav
    Modality-Channel-Emotion-Intensity-Statement-Repetition-Actor

    Emotion codes (position 3):
    01 = neutral, 02 = calm, 03 = happy, 04 = sad,
    05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
    """
    print(f"\n📁 Parsing RAVDESS from: {ravdess_dir}")

    data = []
    ravdess_emotions = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }

    audio_files = list(Path(ravdess_dir).rglob('*.wav'))
    print(f"   Found {len(audio_files)} .wav files")

    for audio_file in audio_files:
        try:
            parts = audio_file.stem.split('-')
            if len(parts) >= 3:
                emotion_code = parts[2]
                emotion = ravdess_emotions.get(emotion_code)

                if emotion:
                    data.append({
                        'filename': str(audio_file),
                        'emotion': emotion,
                        'dataset': 'RAVDESS',
                        **{col: val for col, val in zip(EMOTION_COLS, EMOTION_MAP[emotion])}
                    })
        except Exception as e:
            print(f"   ⚠️  Error parsing {audio_file.name}: {e}")

    df = pd.DataFrame(data)
    print(f"   ✅ Parsed {len(df)} samples")
    return df

# --------------------------
# TESS Parser
# --------------------------
def parse_tess(tess_dir):
    """
    TESS filename format: OAF_back_angry.wav or YAF_dog_sad.wav
    Format: Speaker_Word_Emotion.wav
    """
    print(f"\n📁 Parsing TESS from: {tess_dir}")

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

    audio_files = list(Path(tess_dir).rglob('*.wav'))
    print(f"   Found {len(audio_files)} .wav files")

    for audio_file in audio_files:
        try:
            # Extract emotion from filename
            emotion_raw = audio_file.stem.split('_')[-1].lower()
            emotion = tess_emotion_map.get(emotion_raw)

            if emotion:
                data.append({
                    'filename': str(audio_file),
                    'emotion': emotion,
                    'dataset': 'TESS',
                    **{col: val for col, val in zip(EMOTION_COLS, EMOTION_MAP[emotion])}
                })
        except Exception as e:
            print(f"   ⚠️  Error parsing {audio_file.name}: {e}")

    df = pd.DataFrame(data)
    print(f"   ✅ Parsed {len(df)} samples")
    return df

# --------------------------
# CREMA-D Parser
# --------------------------
def parse_crema(crema_dir):
    """
    CREMA-D filename format: 1001_DFA_ANG_XX.wav
    Format: SpeakerID_Sentence_Emotion_Level.wav
    """
    print(f"\n📁 Parsing CREMA-D from: {crema_dir}")

    data = []
    crema_emotion_map = {
        'ANG': 'angry',
        'DIS': 'disgust',
        'FEA': 'fearful',
        'HAP': 'happy',
        'NEU': 'neutral',
        'SAD': 'sad'
    }

    audio_files = list(Path(crema_dir).rglob('*.wav'))
    print(f"   Found {len(audio_files)} .wav files")

    for audio_file in audio_files:
        try:
            parts = audio_file.stem.split('_')
            if len(parts) >= 3:
                emotion_code = parts[2]
                emotion = crema_emotion_map.get(emotion_code)

                if emotion:
                    data.append({
                        'filename': str(audio_file),
                        'emotion': emotion,
                        'dataset': 'CREMA-D',
                        **{col: val for col, val in zip(EMOTION_COLS, EMOTION_MAP[emotion])}
                    })
        except Exception as e:
            print(f"   ⚠️  Error parsing {audio_file.name}: {e}")

    df = pd.DataFrame(data)
    print(f"   ✅ Parsed {len(df)} samples")
    return df

# --------------------------
# Main Processing Function
# --------------------------
def prepare_eva_dataset(datasets, output_dir=OUTPUT_DIR):
    """
    Combine all datasets and prepare for training
    """
    print("="*70)
    print("🎯 EVA DATASET PREPARATION")
    print("="*70)

    all_data = []

    # Parse RAVDESS
    if datasets.get('ravdess_dir') and os.path.exists(datasets['ravdess_dir']):
        ravdess_df = parse_ravdess(datasets['ravdess_dir'])
        all_data.append(ravdess_df)
    else:
        print(f"\n⚠️  RAVDESS not found at: {datasets.get('ravdess_dir')}")

    # Parse TESS
    if datasets.get('tess_dir') and os.path.exists(datasets['tess_dir']):
        tess_df = parse_tess(datasets['tess_dir'])
        all_data.append(tess_df)
    else:
        print(f"\n⚠️  TESS not found at: {datasets.get('tess_dir')}")

    # Parse CREMA-D
    if datasets.get('crema_dir') and os.path.exists(datasets['crema_dir']):
        crema_df = parse_crema(datasets['crema_dir'])
        all_data.append(crema_df)
    else:
        print(f"\n⚠️  CREMA-D not found at: {datasets.get('crema_dir')}")

    if not all_data:
        raise ValueError("❌ No datasets found! Please check your paths.")

    # Combine all datasets
    combined_df = pd.concat(all_data, ignore_index=True)

    print("\n" + "="*70)
    print(f"📊 COMBINED DATASET STATISTICS")
    print("="*70)
    print(f"Total samples: {len(combined_df)}")
    print(f"\nDataset distribution:")
    dataset_counts = combined_df['dataset'].value_counts()
    for dataset, count in dataset_counts.items():
        print(f"  {dataset:12s}: {count:5d} ({count/len(combined_df)*100:.1f}%)")

    print(f"\nEmotion distribution:")
    emotion_counts = combined_df['emotion'].value_counts().sort_index()
    for emotion, count in emotion_counts.items():
        bar = "█" * int(count / 100)
        print(f"  {emotion:12s}: {count:5d} ({count/len(combined_df)*100:.1f}%) {bar}")

    # Check for missing emotions
    missing_emotions = set(EMOTION_MAP.keys()) - set(emotion_counts.index)
    if missing_emotions:
        print(f"\n⚠️  Missing emotions: {missing_emotions}")
        print("   Note: 'calm' is only in RAVDESS, 'surprised' may be limited")

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/processed_audio", exist_ok=True)
    os.makedirs(f"{output_dir}/labels", exist_ok=True)

    print(f"\n📋 Copying audio files to {output_dir}/processed_audio/...")

    # Copy files with unique naming
    file_counter = {}
    for idx, row in combined_df.iterrows():
        src = row['filename']
        base_name = Path(src).name

        # Add counter to avoid duplicates
        if base_name in file_counter:
            file_counter[base_name] += 1
            new_name = f"{Path(base_name).stem}_{file_counter[base_name]}{Path(base_name).suffix}"
        else:
            file_counter[base_name] = 0
            new_name = base_name

        dst = f"{output_dir}/processed_audio/{new_name}"
        shutil.copy2(src, dst)
        combined_df.at[idx, 'filename'] = new_name

        if (idx + 1) % 1000 == 0:
            print(f"   Copied {idx + 1}/{len(combined_df)} files...")

    print(f"   ✅ Copied all {len(combined_df)} files")

    # Split train/val/test (70/15/15)
    print(f"\n📊 Splitting dataset (70% train, 15% val, 15% test)...")

    train_df, temp_df = train_test_split(
        combined_df,
        test_size=0.3,
        stratify=combined_df['emotion'],
        random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df['emotion'],
        random_state=42
    )

    # Save label files
    label_cols = ['filename'] + EMOTION_COLS
    train_df[label_cols].to_csv(f"{output_dir}/labels/train_labels.csv", index=False)
    val_df[label_cols].to_csv(f"{output_dir}/labels/val_labels.csv", index=False)
    test_df[label_cols].to_csv(f"{output_dir}/labels/test_labels.csv", index=False)

    print(f"   ✅ Train set: {len(train_df)} samples → {output_dir}/labels/train_labels.csv")
    print(f"   ✅ Val set:   {len(val_df)} samples → {output_dir}/labels/val_labels.csv")
    print(f"   ✅ Test set:  {len(test_df)} samples → {output_dir}/labels/test_labels.csv")

    # Save dataset info
    info = {
        'total_samples': len(combined_df),
        'train_samples': len(train_df),
        'val_samples': len(val_df),
        'test_samples': len(test_df),
        'datasets': dataset_counts.to_dict(),
        'emotions': emotion_counts.to_dict(),
        'emotion_labels': EMOTION_COLS
    }

    import json
    with open(f"{output_dir}/dataset_info.json", 'w') as f:
        json.dump(info, f, indent=2)

    print(f"\n📄 Dataset info saved to: {output_dir}/dataset_info.json")

    print("\n" + "="*70)
    print("✅ DATASET PREPARATION COMPLETE!")
    print("="*70)
    print(f"\n🚀 Next steps:")
    print(f"   1. Review the dataset: cat {output_dir}/dataset_info.json")
    print(f"   2. Start training: python train.py")
    print(f"   3. Monitor training: tensorboard --logdir logs/")

    return combined_df, train_df, val_df, test_df

# --------------------------
# Run the script
# --------------------------
if __name__ == "__main__":
    combined_df, train_df, val_df, test_df = prepare_eva_dataset(DATASETS)
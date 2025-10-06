import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
import pandas as pd
import os


class EmotionDataset(Dataset):
    def __init__(self, audio_dir, label_file, sr=16000, n_mels=128, duration=3,
                 hop_length=512, n_fft=2048, augment=False):
        """
        audio_dir: directory containing .wav files
        label_file: CSV file with labels (filename, emotion1, emotion2, ...)
        augment: whether to apply data augmentation (use True for training only)
        """
        self.audio_dir = audio_dir
        self.labels_df = pd.read_csv(label_file)
        self.sr = sr
        self.n_mels = n_mels
        self.duration = duration
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.augment = augment

    def __len__(self):
        return len(self.labels_df)

    def time_shift(self, y, shift_max=0.2):
        """Shift audio in time"""
        shift = int(np.random.uniform(-shift_max, shift_max) * len(y))
        return np.roll(y, shift)

    def add_noise(self, y, noise_factor=0.005):
        """Add random noise"""
        noise = np.random.randn(len(y))
        return y + noise_factor * noise

    def change_pitch(self, y, sr, n_steps=2):
        """Change pitch randomly"""
        n_steps = np.random.randint(-n_steps, n_steps)
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)

    def change_speed(self, y, speed_factor=1.2):
        """Change speed randomly"""
        speed = np.random.uniform(1 / speed_factor, speed_factor)
        return librosa.effects.time_stretch(y, rate=speed)

    def time_mask(self, mel, max_mask_pct=0.1):
        """Apply time masking (SpecAugment)"""
        _, time_steps = mel.shape
        max_mask_frames = int(max_mask_pct * time_steps)

        if max_mask_frames > 0:
            mask_size = np.random.randint(1, max_mask_frames)
            mask_start = np.random.randint(0, time_steps - mask_size)
            mel[:, mask_start:mask_start + mask_size] = 0

        return mel

    def freq_mask(self, mel, max_mask_pct=0.1):
        """Apply frequency masking (SpecAugment)"""
        freq_bins, _ = mel.shape
        max_mask_bins = int(max_mask_pct * freq_bins)

        if max_mask_bins > 0:
            mask_size = np.random.randint(1, max_mask_bins)
            mask_start = np.random.randint(0, freq_bins - mask_size)
            mel[mask_start:mask_start + mask_size, :] = 0

        return mel

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        file_name = row["filename"]
        file_path = os.path.join(self.audio_dir, file_name)

        # ---- Load audio ----
        y, sr = librosa.load(file_path, sr=self.sr, mono=True)

        # ---- Apply audio augmentations (with probability) ----
        if self.augment:
            if np.random.random() < 0.5:
                y = self.time_shift(y)
            if np.random.random() < 0.3:
                y = self.add_noise(y)
            if np.random.random() < 0.3:
                y = self.change_speed(y)
            # Pitch shifting is expensive, use sparingly
            # if np.random.random() < 0.2:
            #     y = self.change_pitch(y, sr)

        # Pad/truncate to fixed length
        max_len = self.duration * self.sr
        if len(y) > max_len:
            # Random crop during training
            if self.augment:
                start = np.random.randint(0, len(y) - max_len)
                y = y[start:start + max_len]
            else:
                y = y[:max_len]
        else:
            y = np.pad(y, (0, max_len - len(y)), mode="constant")

        # ---- Mel Spectrogram ----
        mel = librosa.feature.melspectrogram(
            y=y, sr=self.sr, n_fft=self.n_fft,
            hop_length=self.hop_length, n_mels=self.n_mels
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # ---- Apply spectrogram augmentations ----
        if self.augment:
            if np.random.random() < 0.5:
                mel_db = self.time_mask(mel_db)
            if np.random.random() < 0.5:
                mel_db = self.freq_mask(mel_db)

        # Normalize to [0, 1]
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)

        # Convert to tensor (1, n_mels, time)
        X = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0)

        # ---- Labels ----
        labels = torch.tensor(row[1:].values.astype(float), dtype=torch.float32)

        return X, labels
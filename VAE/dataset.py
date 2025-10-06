import torch
from torch.utils.data import Dataset
import librosa
import numpy as np
import pandas as pd
import os

class EmotionDataset(Dataset):
    def __init__(self, audio_dir, label_file, sr=16000, n_mels=128, duration=3, hop_length=512, n_fft=2048):
        """
        audio_dir: thư mục chứa các file .wav
        label_file: file CSV chứa nhãn (filename, emotion1, emotion2, ...)
        """
        self.audio_dir = audio_dir
        self.labels_df = pd.read_csv(label_file)
        self.sr = sr
        self.n_mels = n_mels
        self.duration = duration
        self.hop_length = hop_length
        self.n_fft = n_fft

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        file_name = row["filename"]
        file_path = os.path.join(self.audio_dir, file_name)

        # ---- Load audio ----
        y, sr = librosa.load(file_path, sr=self.sr, mono=True)

        # Cắt/pad cho đủ độ dài cố định
        max_len = self.duration * self.sr
        if len(y) > max_len:
            y = y[:max_len]
        else:
            y = np.pad(y, (0, max_len - len(y)), mode="constant")

        # ---- Mel Spectrogram ----
        # FIX: Use 'y=' to explicitly name the parameter
        mel = librosa.feature.melspectrogram(
            y=y,                    # Changed from positional to keyword argument
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Chuẩn hóa về [0,1]
        mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-6)

        # Đưa vào tensor (1, n_mels, time)
        X = torch.tensor(mel_norm, dtype=torch.float32).unsqueeze(0)

        # ---- Labels ----
        labels = torch.tensor(row[1:].values.astype(float), dtype=torch.float32)  # multi-label vector

        return X, labels
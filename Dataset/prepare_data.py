import os
import subprocess
import librosa
import soundfile as sf
import numpy as np
import pandas as pd

RAW_DIR = "Dataset/raw_downloads"
PROC_DIR = "Dataset/processed_audio"
LABELS_DIR = "Dataset/labels"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)
os.makedirs(LABELS_DIR, exist_ok=True)

# --------------------------
# 1. Download audio từ YouTube
# --------------------------
def download_youtube_audio(url, out_dir=RAW_DIR):
    cmd = [
        "yt-dlp", "-f", "bestaudio",
        "-x", "--audio-format", "wav",
        "-o", f"{out_dir}/%(id)s.%(ext)s",
        url
    ]
    subprocess.run(cmd)
    print(">> Downloaded:", url)

# --------------------------
# 2. Chuẩn hóa audio (16kHz, mono)
# --------------------------
def load_and_resample(path, sr=16000):
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y, sr

# --------------------------
# 3. Cắt thành chunk dựa trên khoảng lặng
# --------------------------
def split_audio(y, sr, min_s=2, max_s=8, top_db=30):
    intervals = librosa.effects.split(y, top_db=top_db)  # tách theo khoảng lặng
    chunks = []
    for start, end in intervals:
        segment = y[start:end]
        dur = librosa.get_duration(y=segment, sr=sr)
        if dur < min_s:
            continue
        # nếu quá dài thì cắt nhỏ
        if dur > max_s:
            n_parts = int(np.ceil(dur / max_s))
            split_len = len(segment) // n_parts
            for i in range(n_parts):
                chunks.append(segment[i*split_len:(i+1)*split_len])
        else:
            chunks.append(segment)
    return chunks

# --------------------------
# 4. Lưu chunk + CSV template
# --------------------------
def process_audio_file(path, video_id, out_dir=PROC_DIR, sr=16000):
    y, sr = load_and_resample(path, sr)
    chunks = split_audio(y, sr)

    rows = []
    for i, chunk in enumerate(chunks, start=1):
        fname = f"{video_id}_chunk{i:04d}.wav"
        fpath = os.path.join(out_dir, fname)
        sf.write(fpath, chunk, sr, subtype="PCM_16")
        rows.append({"filename": fname})

    return rows

def generate_labels_template(rows, out_csv):
    df = pd.DataFrame(rows)
    emotions = ["happy","sad","angry","neutral","calm","fearful","disgust","surprised"]
    for emo in emotions:
        df[emo] = 0
    df.to_csv(out_csv, index=False)
    print(">> Labels template saved:", out_csv)

# --------------------------
# 5. Main pipeline
# --------------------------
def prepare_dataset(youtube_urls):
    all_rows = []
    for url in youtube_urls:
        # download
        download_youtube_audio(url)
        # lấy id video từ URL (sau dấu =)
        video_id = url.split("v=")[-1]
        raw_path = os.path.join(RAW_DIR, f"{video_id}.wav")

        # process audio
        rows = process_audio_file(raw_path, video_id)
        all_rows.extend(rows)

    # save template labels
    out_csv = os.path.join(LABELS_DIR, "labels_template.csv")
    generate_labels_template(all_rows, out_csv)

if __name__ == "__main__":
    # Danh sách URL YouTube
    urls = [
        "https://www.youtube.com/watch?v=YOUR_VIDEO_ID1",
        "https://www.youtube.com/watch?v=YOUR_VIDEO_ID2"
    ]
    prepare_dataset(urls)

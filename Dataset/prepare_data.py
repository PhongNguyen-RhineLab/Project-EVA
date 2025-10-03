import os
import subprocess
import librosa
import soundfile as sf
import numpy as np
import pandas as pd
import re

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

        # Sử dụng hàm extract_youtube_id để lấy video_id
        video_id = extract_youtube_id(url)

        raw_path = os.path.join(RAW_DIR, f"{video_id}.wav")

        # Check if file exists before processing
        if not os.path.exists(raw_path):
            print(f"Warning: Downloaded file not found at {raw_path}")
            continue

        # process audio
        rows = process_audio_file(raw_path, video_id)
        all_rows.extend(rows)

    # save template labels
    out_csv = os.path.join(LABELS_DIR, "labels_template.csv")
    generate_labels_template(all_rows, out_csv)

def extract_youtube_id(url):
    """
    Trích xuất video_id từ nhiều dạng URL YouTube khác nhau.
    """
    # Các pattern phổ biến của YouTube
    patterns = [
        r"(?:v=|\/videos\/|embed\/|youtu\.be\/|\/shorts\/)([A-Za-z0-9_-]{11})",
        r"youtube\.com\/watch\?.*?v=([A-Za-z0-9_-]{11})",
        r"youtube\.com\/shorts\/([A-Za-z0-9_-]{11})",
        r"youtu\.be\/([A-Za-z0-9_-]{11})"
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    # fallback - use a hash of the URL
    import hashlib
    return hashlib.md5(url.encode()).hexdigest()[:11]

if __name__ == "__main__":
    # Danh sách URL YouTube
    urls = [
        "https://youtube.com/shorts/iph2e7fwB9M?si=yaOISLd1fMm_OI3f",
        "https://youtube.com/shorts/SNqfINV4_8I?si=c-dfPsHjYgsCXO7P",
        "https://youtube.com/shorts/v4ZLu4FfkhE?si=rIM4q5CuyBkoTzno",
        "https://youtube.com/shorts/vMj4om6wr2s?si=oa_4-F_QteZ8mzjK",
        "https://youtube.com/shorts/99aU7NAGvfk?si=tj8Y0MWztMXlu9HY",
        "https://youtube.com/shorts/p3RnM-_Ry_o?si=9QSyHyGFo8GESYj2",
        "https://youtube.com/shorts/cRU8U2-5Yd8?si=c03mDqJZODTM3410",
        "https://youtube.com/shorts/6QANmTjo8JE?si=bJTrIDq-5avCb4U7",
        "https://youtube.com/shorts/ftoFLOg3HKw?si=q3VShACIcEzQHDDP",
        "https://youtube.com/shorts/oFtWhnjwyV8?si=9sJs5b3Fj51RDFt3",
        "https://youtube.com/shorts/uGwTE0issZE?si=85V_VzdoYFMqAUat",
    ]
    prepare_dataset(urls)

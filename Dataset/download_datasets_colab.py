"""
Simple Dataset Downloader for Google Colab
No Kaggle authentication needed!
Uses direct links and Hugging Face datasets
"""

import os
import subprocess
import zipfile
import tarfile
from pathlib import Path
from tqdm import tqdm
import urllib.request

# --------------------------
# Configuration
# --------------------------
OUTPUT_BASE = 'Dataset/prelabel_en'


# --------------------------
# Download Progress Bar
# --------------------------
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path):
    """Download file with progress bar"""
    print(f"⬇️  Downloading to {output_path}...")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc="Progress") as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


# --------------------------
# Extract Archives
# --------------------------
def extract_archive(archive_path, extract_to):
    """Extract zip or tar archives"""
    print(f"📦 Extracting {archive_path}...")

    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            # Extract with progress
            members = zip_ref.namelist()
            for i, member in enumerate(tqdm(members, desc="Extracting")):
                zip_ref.extract(member, extract_to)
    elif archive_path.endswith(('.tar.gz', '.tgz')):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)

    print(f"✅ Extracted to {extract_to}")


# --------------------------
# 1. RAVDESS from Zenodo (Direct Link)
# --------------------------
def download_ravdess():
    """Download RAVDESS - No authentication needed"""
    print("\n" + "=" * 70)
    print("📥 RAVDESS - Ryerson Audio-Visual Database")
    print("=" * 70)

    output_dir = f"{OUTPUT_BASE}/RAVDESS"
    os.makedirs(output_dir, exist_ok=True)

    # Check if already downloaded
    if os.path.exists(f"{output_dir}/Audio_Speech_Actors_01-24"):
        wav_count = len(list(Path(output_dir).rglob('*.wav')))
        if wav_count > 7000:
            print(f"✅ RAVDESS already downloaded ({wav_count} files)")
            return True

    zip_path = f"{output_dir}/ravdess.zip"

    try:
        # Direct Zenodo link
        url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
        download_file(url, zip_path)
        extract_archive(zip_path, output_dir)
        os.remove(zip_path)
        print("✅ RAVDESS download complete!")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


# --------------------------
# 2. TESS from Hugging Face (No Kaggle needed!)
# --------------------------
def download_tess_huggingface():
    """Download TESS from Hugging Face - No authentication needed"""
    print("\n" + "=" * 70)
    print("📥 TESS - Toronto Emotional Speech Set")
    print("=" * 70)

    output_dir = f"{OUTPUT_BASE}/TESS"
    os.makedirs(output_dir, exist_ok=True)

    # Check if already downloaded
    wav_count = len(list(Path(output_dir).rglob('*.wav')))
    if wav_count > 2500:
        print(f"✅ TESS already downloaded ({wav_count} files)")
        return True

    try:
        # Install datasets library if needed
        subprocess.run(["pip", "install", "-q", "datasets"], check=True)

        from datasets import load_dataset

        print("⬇️  Downloading TESS from Hugging Face...")

        # Load TESS dataset from Hugging Face
        dataset = load_dataset("DynamicSuperb/EmotionRecognition_TESS", split="test")

        print(f"📊 Loaded {len(dataset)} samples")

        # Organize by emotion
        emotions = {}
        for item in tqdm(dataset, desc="Processing"):
            audio = item['audio']
            label = item['label']

            if label not in emotions:
                emotions[label] = []
            emotions[label].append(audio)

        # Save audio files organized by emotion
        file_counter = 0
        for emotion, audios in emotions.items():
            emotion_dir = f"{output_dir}/{emotion}"
            os.makedirs(emotion_dir, exist_ok=True)

            for i, audio in enumerate(tqdm(audios, desc=f"Saving {emotion}")):
                import soundfile as sf
                output_file = f"{emotion_dir}/tess_{emotion}_{i:04d}.wav"
                sf.write(output_file, audio['array'], audio['sampling_rate'])
                file_counter += 1

        print(f"✅ TESS download complete! ({file_counter} files)")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n🔄 Trying alternative method...")
        return download_tess_gdrive()


# --------------------------
# 3. TESS from Google Drive (Alternative)
# --------------------------
def download_tess_gdrive():
    """Download TESS from Google Drive mirror"""
    print("\n📥 Downloading TESS from Google Drive mirror...")

    output_dir = f"{OUTPUT_BASE}/TESS"
    os.makedirs(output_dir, exist_ok=True)

    try:
        import gdown

        # Public Google Drive link for TESS (if available)
        # Note: This is an example ID - replace with actual public link
        gdrive_id = "1VxR5L0-pCxSjLpNvvJH4ppD_4yJq3gKL"
        zip_path = f"{output_dir}/tess.zip"

        print("⬇️  Downloading from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={gdrive_id}", zip_path, quiet=False)

        extract_archive(zip_path, output_dir)
        os.remove(zip_path)

        print("✅ TESS download complete!")
        return True

    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n⚠️  Manual download required for TESS")
        print("Visit: https://tspace.library.utoronto.ca/handle/1807/24487")
        return False


# --------------------------
# 4. CREMA-D from TalkBank (Direct Link)
# --------------------------
def download_crema():
    """Download CREMA-D from TalkBank - No authentication needed"""
    print("\n" + "=" * 70)
    print("📥 CREMA-D - Crowd Sourced Emotional Multimodal Actors")
    print("=" * 70)

    output_dir = f"{OUTPUT_BASE}/CREMA-D"
    os.makedirs(output_dir, exist_ok=True)

    # Check if already downloaded
    wav_count = len(list(Path(output_dir).rglob('*.wav')))
    if wav_count > 7000:
        print(f"✅ CREMA-D already downloaded ({wav_count} files)")
        return True

    zip_path = f"{output_dir}/AudioWAV.zip"

    try:
        # Direct TalkBank link
        url = "https://media.talkbank.org/ca/CREMA/AudioWAV.zip"
        print("⬇️  Downloading CREMA-D (~2GB, this will take a while)...")
        download_file(url, zip_path)
        extract_archive(zip_path, output_dir)
        os.remove(zip_path)
        print("✅ CREMA-D download complete!")
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


# --------------------------
# Main Download Function
# --------------------------
def download_all():
    """Download all datasets - Colab friendly!"""
    print("=" * 70)
    print("🎯 EVA PROJECT - SIMPLE DATASET DOWNLOADER")
    print("   (Google Colab Compatible - No Kaggle Auth Required!)")
    print("=" * 70)
    print("\nThis will download:")
    print("  1. RAVDESS from Zenodo (~1.5GB)")
    print("  2. TESS from Hugging Face (~400MB)")
    print("  3. CREMA-D from TalkBank (~2GB)")
    print("\nTotal: ~4GB, ~17,600 audio samples")

    # Create base directory
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    results = {}

    # Download each dataset
    print("\n🚀 Starting downloads...\n")

    results['RAVDESS'] = download_ravdess()
    results['TESS'] = download_tess_huggingface()
    results['CREMA-D'] = download_crema()

    # Summary
    print("\n" + "=" * 70)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 70)

    for dataset, success in results.items():
        status = "✅ SUCCESS" if success else "⚠️  NEEDS MANUAL DOWNLOAD"
        print(f"  {dataset:12s}: {status}")

    successful = sum(results.values())
    print(f"\n  Total: {successful}/{len(results)} datasets downloaded")

    # Verify
    print("\n" + "=" * 70)
    print("📁 VERIFICATION")
    print("=" * 70)

    total_files = 0
    for dataset_name in ['RAVDESS', 'TESS', 'CREMA-D']:
        dataset_dir = f"{OUTPUT_BASE}/{dataset_name}"
        if os.path.exists(dataset_dir):
            wav_files = list(Path(dataset_dir).rglob('*.wav'))
            count = len(wav_files)
            total_files += count
            print(f"  {dataset_name:12s}: {count:5d} .wav files")
        else:
            print(f"  {dataset_name:12s}: ⚠️  Not found")

    print(f"\n  Total audio files: {total_files}")

    if successful >= 2:
        print("\n" + "=" * 70)
        print("✅ READY FOR NEXT STEP!")
        print("=" * 70)
        print("\n🚀 Run next:")
        print("   python setup_eva_dataset.py")

    return results


# --------------------------
# Google Colab Auto-Install
# --------------------------
def setup_colab_env():
    """Auto-install dependencies for Colab"""
    print("🔧 Setting up Colab environment...\n")

    packages = [
        "datasets",
        "soundfile",
        "gdown",
        "librosa",
        "pandas",
        "scikit-learn",
        "tqdm"
    ]

    for package in packages:
        try:
            __import__(package)
        except ImportError:
            print(f"Installing {package}...")
            subprocess.run(["pip", "install", "-q", package], check=True)

    print("✅ Environment ready!\n")


# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    import sys

    # Check if running in Colab
    try:
        import google.colab

        IN_COLAB = True
        print("🔬 Google Colab detected!")
        setup_colab_env()
    except:
        IN_COLAB = False

    # Parse arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == '--ravdess':
            download_ravdess()
        elif sys.argv[1] == '--tess':
            download_tess_huggingface()
        elif sys.argv[1] == '--crema':
            download_crema()
        else:
            print("Usage:")
            print("  python download_datasets_colab.py              # Download all")
            print("  python download_datasets_colab.py --ravdess    # RAVDESS only")
            print("  python download_datasets_colab.py --tess       # TESS only")
            print("  python download_datasets_colab.py --crema      # CREMA-D only")
    else:
        download_all()
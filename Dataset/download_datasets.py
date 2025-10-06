"""
Automatic Dataset Downloader for EVA Project
Downloads RAVDESS, TESS, and CREMA-D datasets
"""

import os
import urllib.request
import zipfile
import tarfile
import requests
from pathlib import Path
from tqdm import tqdm
import gdown

# --------------------------
# Configuration
# --------------------------
OUTPUT_BASE = 'Dataset/prelabel_en'

DATASETS = {
    'RAVDESS': {
        'url': 'https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip',
        'output_dir': f'{OUTPUT_BASE}/RAVDESS',
        'type': 'zip',
        'description': 'RAVDESS - Ryerson Audio-Visual Database of Emotional Speech and Song'
    },
    'TESS': {
        'url': 'https://www.kaggle.com/api/v1/datasets/download/ejlok1/toronto-emotional-speech-set-tess',
        'output_dir': f'{OUTPUT_BASE}/TESS',
        'type': 'zip',
        'description': 'TESS - Toronto Emotional Speech Set',
        'requires_kaggle': True
    },
    'CREMA-D': {
        'url': 'https://github.com/CheyneyComputerScience/CREMA-D/archive/refs/heads/master.zip',
        'output_dir': f'{OUTPUT_BASE}/CREMA-D',
        'type': 'zip',
        'description': 'CREMA-D - Crowd Sourced Emotional Multimodal Actors Dataset',
        'note': 'Large dataset (~2GB), may take time'
    }
}


# --------------------------
# Download with Progress Bar
# --------------------------
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url, output_path):
    """Download file with progress bar"""
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=output_path) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def download_with_requests(url, output_path):
    """Alternative download method with requests"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(output_path, 'wb') as f, tqdm(
            desc=output_path,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            bar.update(size)


# --------------------------
# Extract Archives
# --------------------------
def extract_archive(archive_path, extract_to):
    """Extract zip or tar archives"""
    print(f"\n📦 Extracting {archive_path}...")

    if archive_path.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.endswith(('.tar.gz', '.tgz')):
        with tarfile.open(archive_path, 'r:gz') as tar_ref:
            tar_ref.extractall(extract_to)
    elif archive_path.endswith('.tar'):
        with tarfile.open(archive_path, 'r') as tar_ref:
            tar_ref.extractall(extract_to)

    print(f"   ✅ Extracted to {extract_to}")


# --------------------------
# RAVDESS Downloader
# --------------------------
def download_ravdess():
    """Download RAVDESS dataset from Zenodo"""
    dataset_info = DATASETS['RAVDESS']
    output_dir = dataset_info['output_dir']

    print(f"\n{'=' * 70}")
    print(f"📥 Downloading RAVDESS")
    print(f"{'=' * 70}")
    print(f"Description: {dataset_info['description']}")
    print(f"URL: {dataset_info['url']}")

    os.makedirs(output_dir, exist_ok=True)

    # Download
    zip_path = f"{output_dir}/ravdess.zip"

    if os.path.exists(zip_path):
        print(f"⚠️  File already exists: {zip_path}")
        response = input("Re-download? (y/n): ")
        if response.lower() != 'y':
            print("Skipping download...")
            if os.path.exists(f"{output_dir}/Audio_Speech_Actors_01-24"):
                print("✅ RAVDESS already downloaded and extracted")
                return True

    try:
        print(f"\n⬇️  Downloading RAVDESS (~1.5GB)...")
        download_file(dataset_info['url'], zip_path)

        # Extract
        extract_archive(zip_path, output_dir)

        # Clean up
        os.remove(zip_path)
        print("🗑️  Removed zip file")

        print("✅ RAVDESS download complete!")
        return True

    except Exception as e:
        print(f"❌ Error downloading RAVDESS: {e}")
        print("\nManual download instructions:")
        print("1. Visit: https://zenodo.org/record/1188976")
        print("2. Download 'Audio_Speech_Actors_01-24.zip'")
        print(f"3. Extract to: {output_dir}/")
        return False


# --------------------------
# TESS Downloader (Kaggle)
# --------------------------
def download_tess():
    """Download TESS dataset from Kaggle"""
    dataset_info = DATASETS['TESS']
    output_dir = dataset_info['output_dir']

    print(f"\n{'=' * 70}")
    print(f"📥 Downloading TESS")
    print(f"{'=' * 70}")
    print(f"Description: {dataset_info['description']}")

    os.makedirs(output_dir, exist_ok=True)

    # Check if kaggle is installed
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        # Authenticate
        api = KaggleApi()
        api.authenticate()

        print(f"\n⬇️  Downloading TESS from Kaggle...")

        # Download using Kaggle API
        api.dataset_download_files(
            'ejlok1/toronto-emotional-speech-set-tess',
            path=output_dir,
            unzip=True
        )

        print("✅ TESS download complete!")
        return True

    except ImportError:
        print("❌ Kaggle API not installed")
        print("\nOption 1 - Install Kaggle API:")
        print("  pip install kaggle")
        print("\nOption 2 - Manual download:")
        print("  1. Visit: https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess")
        print("  2. Click 'Download'")
        print(f"  3. Extract to: {output_dir}/")
        print("\nOption 3 - Alternative direct link:")

        # Try alternative Google Drive link (TESS is also on Drive)
        try_google_drive_tess(output_dir)
        return False

    except Exception as e:
        print(f"❌ Error downloading TESS: {e}")
        print("\nManual download instructions:")
        print("1. Create Kaggle account: https://www.kaggle.com/")
        print("2. Download dataset: https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess")
        print(f"3. Extract to: {output_dir}/")
        return False


def try_google_drive_tess(output_dir):
    """Try downloading TESS from Google Drive alternative"""
    # TESS Google Drive ID (public shared link)
    # Note: This may change, update if needed
    gdrive_id = "1VmZWZhpKJWJLlqm8-rdrXlR4hDHCqWyc"  # Example ID

    print("\n🔄 Trying Google Drive alternative...")
    try:
        import gdown
        output_path = f"{output_dir}/tess.zip"
        gdown.download(f"https://drive.google.com/uc?id={gdrive_id}", output_path, quiet=False)
        extract_archive(output_path, output_dir)
        os.remove(output_path)
        print("✅ TESS downloaded from Google Drive!")
        return True
    except:
        print("❌ Google Drive download failed")
        return False


# --------------------------
# CREMA-D Downloader
# --------------------------
def download_crema_d():
    """Download CREMA-D dataset"""
    dataset_info = DATASETS['CREMA-D']
    output_dir = dataset_info['output_dir']

    print(f"\n{'=' * 70}")
    print(f"📥 Downloading CREMA-D")
    print(f"{'=' * 70}")
    print(f"Description: {dataset_info['description']}")
    print(f"Note: {dataset_info.get('note', '')}")

    os.makedirs(output_dir, exist_ok=True)

    # CREMA-D needs special handling - audio files are large
    # GitHub repo only has metadata, actual audio is on external server

    print("\n⚠️  CREMA-D requires manual download")
    print("The dataset is ~2GB and hosted externally")
    print("\nDownload instructions:")
    print("1. Visit: https://github.com/CheyneyComputerScience/CREMA-D")
    print("2. Follow instructions to download audio files")
    print("3. Download link: https://media.talkbank.org/ca/CREMA/AudioWAV.zip")
    print(f"4. Extract WAV files to: {output_dir}/")

    # Try direct download
    response = input("\nTry direct download from TalkBank? (y/n): ")
    if response.lower() == 'y':
        try:
            crema_url = "https://media.talkbank.org/ca/CREMA/AudioWAV.zip"
            zip_path = f"{output_dir}/AudioWAV.zip"

            print(f"\n⬇️  Downloading CREMA-D (~2GB, this may take a while)...")
            download_with_requests(crema_url, zip_path)

            extract_archive(zip_path, output_dir)
            os.remove(zip_path)

            print("✅ CREMA-D download complete!")
            return True

        except Exception as e:
            print(f"❌ Error downloading CREMA-D: {e}")
            return False

    return False


# --------------------------
# Main Download Function
# --------------------------
def download_all_datasets():
    """Download all datasets"""
    print("=" * 70)
    print("🎯 EVA PROJECT - DATASET DOWNLOADER")
    print("=" * 70)
    print("\nThis script will download:")
    print("  1. RAVDESS (~1.5GB) - 7,356 samples")
    print("  2. TESS (~400MB) - 2,800 samples")
    print("  3. CREMA-D (~2GB) - 7,442 samples")
    print("\nTotal download size: ~4GB")
    print("Total samples: ~17,600")

    response = input("\nProceed with download? (y/n): ")
    if response.lower() != 'y':
        print("Download cancelled")
        return

    # Create base directory
    os.makedirs(OUTPUT_BASE, exist_ok=True)

    results = {}

    # Download RAVDESS
    results['RAVDESS'] = download_ravdess()

    # Download TESS
    results['TESS'] = download_tess()

    # Download CREMA-D
    results['CREMA-D'] = download_crema_d()

    # Summary
    print("\n" + "=" * 70)
    print("📊 DOWNLOAD SUMMARY")
    print("=" * 70)

    for dataset, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED (manual download needed)"
        print(f"{dataset:15s}: {status}")

    successful = sum(results.values())
    print(f"\nTotal: {successful}/{len(results)} datasets downloaded successfully")

    if successful > 0:
        print(f"\n🚀 Next step:")
        print(f"   python setup_eva_dataset.py")

    # Verify downloads
    print("\n" + "=" * 70)
    print("📁 VERIFICATION")
    print("=" * 70)

    for dataset_name, dataset_info in DATASETS.items():
        output_dir = dataset_info['output_dir']
        if os.path.exists(output_dir):
            wav_files = list(Path(output_dir).rglob('*.wav'))
            print(f"{dataset_name:15s}: {len(wav_files):5d} .wav files found in {output_dir}")
        else:
            print(f"{dataset_name:15s}: ⚠️  Directory not found: {output_dir}")


# --------------------------
# Setup Kaggle (Helper Function)
# --------------------------
def setup_kaggle():
    """Help user setup Kaggle API credentials"""
    print("\n" + "=" * 70)
    print("🔧 KAGGLE API SETUP")
    print("=" * 70)
    print("\nTo download TESS from Kaggle, you need to setup Kaggle API:")
    print("\n1. Install Kaggle API:")
    print("   pip install kaggle")
    print("\n2. Get your Kaggle API credentials:")
    print("   a. Go to: https://www.kaggle.com/")
    print("   b. Login and go to Account settings")
    print("   c. Scroll to 'API' section")
    print("   d. Click 'Create New API Token'")
    print("   e. Download kaggle.json")
    print("\n3. Place kaggle.json in:")
    print("   Linux/Mac: ~/.kaggle/kaggle.json")
    print("   Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json")
    print("\n4. Set permissions (Linux/Mac only):")
    print("   chmod 600 ~/.kaggle/kaggle.json")
    print("\n5. Run this script again")


# --------------------------
# Main Execution
# --------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == '--setup-kaggle':
            setup_kaggle()
        elif sys.argv[1] == '--ravdess':
            download_ravdess()
        elif sys.argv[1] == '--tess':
            download_tess()
        elif sys.argv[1] == '--crema':
            download_crema_d()
        else:
            print("Usage:")
            print("  python download_datasets.py              # Download all")
            print("  python download_datasets.py --ravdess    # Download RAVDESS only")
            print("  python download_datasets.py --tess       # Download TESS only")
            print("  python download_datasets.py --crema      # Download CREMA-D only")
            print("  python download_datasets.py --setup-kaggle  # Kaggle setup help")
    else:
        download_all_datasets()
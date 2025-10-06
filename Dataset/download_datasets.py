"""
Robust Dataset Downloader for EVA Project
Handles all edge cases and provides clear error messages
"""

import os
import sys
import urllib.request
import zipfile
import tarfile
import requests
from pathlib import Path
from tqdm import tqdm
import time
import hashlib

# Configuration
OUTPUT_BASE = 'Dataset/prelabel_en'
TEMP_DIR = 'Dataset/temp'

class Colors:
    """ANSI color codes for pretty printing"""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ {text}{Colors.RESET}")


class DownloadProgressBar(tqdm):
    """Progress bar for downloads"""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def check_disk_space(required_gb=5):
    """Check if enough disk space available"""
    try:
        import shutil
        stat = shutil.disk_usage('.')
        free_gb = stat.free / (1024**3)

        if free_gb < required_gb:
            print_warning(f"Low disk space: {free_gb:.1f}GB free (need ~{required_gb}GB)")
            return False
        return True
    except:
        return True  # Skip check if unable to determine


def download_file(url, output_path, timeout=300, max_retries=3):
    """
    Download file with retry logic and proper error handling
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for attempt in range(max_retries):
        try:
            print_info(f"Downloading from: {url}")
            print_info(f"Saving to: {output_path}")

            # Try with requests first (better error handling)
            response = requests.get(url, stream=True, timeout=timeout)
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))

            with open(output_path, 'wb') as f, tqdm(
                desc="Downloading",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    size = f.write(chunk)
                    pbar.update(size)

            print_success(f"Downloaded successfully!")
            return True

        except requests.exceptions.Timeout:
            print_error(f"Timeout error (attempt {attempt+1}/{max_retries})")
            if attempt < max_retries - 1:
                print_info("Retrying in 5 seconds...")
                time.sleep(5)

        except requests.exceptions.ConnectionError:
            print_error(f"Connection error (attempt {attempt+1}/{max_retries})")
            if attempt < max_retries - 1:
                print_info("Retrying in 5 seconds...")
                time.sleep(5)

        except requests.exceptions.HTTPError as e:
            print_error(f"HTTP Error: {e}")
            return False

        except Exception as e:
            print_error(f"Unexpected error: {e}")
            if attempt < max_retries - 1:
                print_info("Retrying...")
                time.sleep(5)

    return False


def extract_archive(archive_path, extract_to, remove_after=True):
    """
    Extract zip/tar archives with progress bar
    """
    print_info(f"Extracting {os.path.basename(archive_path)}...")

    try:
        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                members = zip_ref.namelist()
                for member in tqdm(members, desc="Extracting"):
                    try:
                        zip_ref.extract(member, extract_to)
                    except:
                        # Skip problematic files
                        continue

        elif archive_path.endswith(('.tar.gz', '.tgz')):
            with tarfile.open(archive_path, 'r:gz') as tar_ref:
                tar_ref.extractall(extract_to)

        elif archive_path.endswith('.tar'):
            with tarfile.open(archive_path, 'r') as tar_ref:
                tar_ref.extractall(extract_to)

        print_success(f"Extracted to {extract_to}")

        # Clean up archive
        if remove_after:
            os.remove(archive_path)
            print_info("Removed archive file")

        return True

    except Exception as e:
        print_error(f"Extraction failed: {e}")
        return False


def verify_dataset(dataset_dir, expected_files=None, min_wav_count=100):
    """
    Verify that dataset was downloaded correctly
    """
    if not os.path.exists(dataset_dir):
        return False, "Directory does not exist"

    wav_files = list(Path(dataset_dir).rglob('*.wav'))
    wav_count = len(wav_files)

    if wav_count < min_wav_count:
        return False, f"Only {wav_count} .wav files found (expected >{min_wav_count})"

    return True, f"Found {wav_count} .wav files"


# ============================================================================
# RAVDESS Downloader
# ============================================================================
def download_ravdess():
    """
    Download RAVDESS from Zenodo (most reliable source)
    """
    print_header("📥 DOWNLOADING RAVDESS")

    output_dir = f"{OUTPUT_BASE}/RAVDESS"
    os.makedirs(output_dir, exist_ok=True)

    # Check if already downloaded
    is_valid, msg = verify_dataset(output_dir, min_wav_count=1400)
    if is_valid:
        print_success(f"RAVDESS already downloaded! ({msg})")
        return True

    print_info("Dataset: Ryerson Audio-Visual Database of Emotional Speech and Song")
    print_info("Size: ~1.5GB | Samples: ~1,440 files")
    print_info("Source: Zenodo (https://zenodo.org/record/1188976)")

    # URL
    url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
    zip_path = f"{TEMP_DIR}/ravdess.zip"

    # Download
    os.makedirs(TEMP_DIR, exist_ok=True)
    success = download_file(url, zip_path)

    if not success:
        print_error("Download failed!")
        print_manual_instructions_ravdess()
        return False

    # Extract
    success = extract_archive(zip_path, output_dir)

    if not success:
        print_error("Extraction failed!")
        return False

    # Verify
    is_valid, msg = verify_dataset(output_dir, min_wav_count=1400)
    if is_valid:
        print_success(f"RAVDESS downloaded successfully! ({msg})")
        return True
    else:
        print_error(f"Verification failed: {msg}")
        return False


def print_manual_instructions_ravdess():
    """Print manual download instructions for RAVDESS"""
    print_warning("\nMANUAL DOWNLOAD INSTRUCTIONS:")
    print("1. Visit: https://zenodo.org/record/1188976")
    print("2. Download: 'Audio_Speech_Actors_01-24.zip' (~1.5GB)")
    print(f"3. Extract to: {OUTPUT_BASE}/RAVDESS/")
    print("4. Run this script again to continue\n")


# ============================================================================
# TESS Downloader (Multiple Methods)
# ============================================================================
def download_tess():
    """
    Download TESS with fallback methods
    """
    print_header("📥 DOWNLOADING TESS")

    output_dir = f"{OUTPUT_BASE}/TESS"
    os.makedirs(output_dir, exist_ok=True)

    # Check if already downloaded
    is_valid, msg = verify_dataset(output_dir, min_wav_count=2500)
    if is_valid:
        print_success(f"TESS already downloaded! ({msg})")
        return True

    print_info("Dataset: Toronto Emotional Speech Set")
    print_info("Size: ~400MB | Samples: ~2,800 files")

    # Method 1: Try Kaggle API
    print_info("\n[Method 1/3] Trying Kaggle API...")
    if download_tess_kaggle(output_dir):
        return True

    # Method 2: Try Hugging Face
    print_info("\n[Method 2/3] Trying Hugging Face...")
    if download_tess_huggingface(output_dir):
        return True

    # Method 3: Manual instructions
    print_warning("\n[Method 3/3] Automatic download failed")
    print_manual_instructions_tess()
    return False


def download_tess_kaggle(output_dir):
    """Try downloading TESS via Kaggle API"""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        print_info("Authenticating with Kaggle...")
        api = KaggleApi()
        api.authenticate()

        print_info("Downloading from Kaggle...")
        api.dataset_download_files(
            'ejlok1/toronto-emotional-speech-set-tess',
            path=output_dir,
            unzip=True
        )

        # Verify
        is_valid, msg = verify_dataset(output_dir, min_wav_count=2500)
        if is_valid:
            print_success(f"TESS downloaded via Kaggle! ({msg})")
            return True

        return False

    except ImportError:
        print_warning("Kaggle API not installed (pip install kaggle)")
        return False
    except Exception as e:
        print_warning(f"Kaggle download failed: {e}")
        return False


def download_tess_huggingface(output_dir):
    """Try downloading TESS via Hugging Face"""
    try:
        from datasets import load_dataset
        import soundfile as sf

        print_info("Loading from Hugging Face...")
        dataset = load_dataset("DynamicSuperb/EmotionRecognition_TESS", split="test")

        print_info(f"Processing {len(dataset)} samples...")

        # Organize by emotion
        emotions = {}
        for item in tqdm(dataset, desc="Organizing"):
            audio = item['audio']
            label = item['label']

            if label not in emotions:
                emotions[label] = []
            emotions[label].append(audio)

        # Save files
        file_count = 0
        for emotion, audios in emotions.items():
            emotion_dir = f"{output_dir}/{emotion}"
            os.makedirs(emotion_dir, exist_ok=True)

            for i, audio in enumerate(tqdm(audios, desc=f"Saving {emotion}")):
                output_file = f"{emotion_dir}/tess_{emotion}_{i:04d}.wav"
                sf.write(output_file, audio['array'], audio['sampling_rate'])
                file_count += 1

        print_success(f"TESS downloaded via Hugging Face! ({file_count} files)")
        return True

    except ImportError as e:
        print_warning(f"Missing library: {e}")
        print_info("Install: pip install datasets soundfile")
        return False
    except Exception as e:
        print_warning(f"Hugging Face download failed: {e}")
        return False


def print_manual_instructions_tess():
    """Print manual download instructions for TESS"""
    print_warning("\nMANUAL DOWNLOAD INSTRUCTIONS:")
    print("\n[Option 1] Kaggle (Recommended):")
    print("  1. Install: pip install kaggle")
    print("  2. Setup API key:")
    print("     - Go to https://www.kaggle.com/settings")
    print("     - Click 'Create New API Token'")
    print("     - Place kaggle.json in ~/.kaggle/")
    print("  3. Run this script again")

    print("\n[Option 2] Manual Download:")
    print("  1. Visit: https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess")
    print("  2. Click 'Download' (requires Kaggle account)")
    print(f"  3. Extract to: {OUTPUT_BASE}/TESS/")

    print("\n[Option 3] University Source:")
    print("  1. Visit: https://tspace.library.utoronto.ca/handle/1807/24487")
    print("  2. Download TESS dataset")
    print(f"  3. Extract to: {OUTPUT_BASE}/TESS/\n")


# ============================================================================
# CREMA-D Downloader
# ============================================================================
def download_crema_d():
    """
    Download CREMA-D from TalkBank
    """
    print_header("📥 DOWNLOADING CREMA-D")

    output_dir = f"{OUTPUT_BASE}/CREMA-D"
    os.makedirs(output_dir, exist_ok=True)

    # Check if already downloaded
    is_valid, msg = verify_dataset(output_dir, min_wav_count=7000)
    if is_valid:
        print_success(f"CREMA-D already downloaded! ({msg})")
        return True

    print_info("Dataset: Crowd Sourced Emotional Multimodal Actors Dataset")
    print_info("Size: ~2GB | Samples: ~7,442 files")
    print_info("Source: TalkBank (https://media.talkbank.org)")
    print_warning("Note: Large file, download may take 10-30 minutes")

    # Ask user confirmation
    response = input("\nProceed with download? [y/N]: ").strip().lower()
    if response != 'y':
        print_info("Skipping CREMA-D download")
        return False

    # URL
    url = "https://media.talkbank.org/ca/CREMA/AudioWAV.zip"
    zip_path = f"{TEMP_DIR}/crema_d.zip"

    # Download
    os.makedirs(TEMP_DIR, exist_ok=True)
    success = download_file(url, zip_path, timeout=600)  # 10min timeout

    if not success:
        print_error("Download failed!")
        print_manual_instructions_crema()
        return False

    # Extract
    success = extract_archive(zip_path, output_dir)

    if not success:
        print_error("Extraction failed!")
        return False

    # Verify
    is_valid, msg = verify_dataset(output_dir, min_wav_count=7000)
    if is_valid:
        print_success(f"CREMA-D downloaded successfully! ({msg})")
        return True
    else:
        print_error(f"Verification failed: {msg}")
        return False


def print_manual_instructions_crema():
    """Print manual download instructions for CREMA-D"""
    print_warning("\nMANUAL DOWNLOAD INSTRUCTIONS:")
    print("1. Visit: https://github.com/CheyneyComputerScience/CREMA-D")
    print("2. Or direct link: https://media.talkbank.org/ca/CREMA/AudioWAV.zip")
    print("3. Download AudioWAV.zip (~2GB)")
    print(f"4. Extract to: {OUTPUT_BASE}/CREMA-D/\n")


# ============================================================================
# Main Function
# ============================================================================
def main():
    """Main download orchestrator"""
    print_header("🎯 EVA PROJECT - DATASET DOWNLOADER")

    print("This script will download:")
    print("  1. RAVDESS  (~1.5GB) - 1,440 samples")
    print("  2. TESS     (~400MB) - 2,800 samples")
    print("  3. CREMA-D  (~2GB)   - 7,442 samples")
    print("\nTotal: ~4GB, ~11,682 audio samples")
    print(f"\nOutput directory: {OUTPUT_BASE}/")

    # Check disk space
    if not check_disk_space(5):
        response = input("\nContinue anyway? [y/N]: ").strip().lower()
        if response != 'y':
            print_info("Cancelled by user")
            return

    # Create directories
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    # Download each dataset
    results = {}

    print_info("\nStarting downloads...\n")

    results['RAVDESS'] = download_ravdess()
    results['TESS'] = download_tess()
    results['CREMA-D'] = download_crema_d()

    # Summary
    print_header("📊 DOWNLOAD SUMMARY")

    for dataset, success in results.items():
        status = f"{Colors.GREEN}✓ SUCCESS{Colors.RESET}" if success else f"{Colors.RED}✗ FAILED{Colors.RESET}"
        print(f"  {dataset:12s}: {status}")

    successful = sum(results.values())
    total = len(results)

    print(f"\n  Total: {successful}/{total} datasets downloaded successfully")

    # Verification
    print_header("📁 VERIFICATION")

    total_files = 0
    for dataset_name in ['RAVDESS', 'TESS', 'CREMA-D']:
        dataset_dir = f"{OUTPUT_BASE}/{dataset_name}"
        if os.path.exists(dataset_dir):
            wav_files = list(Path(dataset_dir).rglob('*.wav'))
            count = len(wav_files)
            total_files += count
            print(f"  {dataset_name:12s}: {count:5d} .wav files")
        else:
            print(f"  {dataset_name:12s}: {Colors.YELLOW}⚠ Not found{Colors.RESET}")

    print(f"\n  Total audio files: {total_files}")

    # Next steps
    if successful >= 2:
        print_header("✅ READY FOR NEXT STEP")
        print("\n🚀 Run next:")
        print(f"   python Dataset/prepare_dataset.py")
    else:
        print_header("⚠ INCOMPLETE DOWNLOAD")
        print("\nPlease complete manual downloads for failed datasets")
        print("Then run this script again to verify")

    # Cleanup temp directory
    try:
        import shutil
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
            print_info(f"\nCleaned up temporary files")
    except:
        pass


# ============================================================================
# CLI Interface
# ============================================================================
if __name__ == "__main__":
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == '--ravdess':
            download_ravdess()
        elif command == '--tess':
            download_tess()
        elif command == '--crema':
            download_crema_d()
        elif command == '--verify':
            # Verify all datasets
            print_header("📁 VERIFYING DATASETS")
            for name in ['RAVDESS', 'TESS', 'CREMA-D']:
                dataset_dir = f"{OUTPUT_BASE}/{name}"
                is_valid, msg = verify_dataset(dataset_dir, min_wav_count=100)
                status = f"{Colors.GREEN}✓{Colors.RESET}" if is_valid else f"{Colors.RED}✗{Colors.RESET}"
                print(f"  {name:12s} {status} {msg}")
        elif command in ['--help', '-h']:
            print("Usage:")
            print("  python download_datasets_robust.py              # Download all")
            print("  python download_datasets_robust.py --ravdess    # RAVDESS only")
            print("  python download_datasets_robust.py --tess       # TESS only")
            print("  python download_datasets_robust.py --crema      # CREMA-D only")
            print("  python download_datasets_robust.py --verify     # Verify downloads")
        else:
            print_error(f"Unknown command: {command}")
            print("Use --help for usage information")
    else:
        main()
"""
Dataset Downloader for EVA Project

Downloads RAVDESS, TESS, and CREMA-D emotion speech datasets.
"""

import os
import sys
import shutil
import zipfile
import tarfile
import requests
from pathlib import Path
from typing import Tuple, Optional

try:
    from tqdm import tqdm
except ImportError:
    # Fallback progress indicator
    def tqdm(iterable, **kwargs):
        return iterable

# Configuration
OUTPUT_BASE = 'Dataset/prelabel_en'
TEMP_DIR = 'Dataset/.temp'

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from console import console, Colors
except ImportError:
    # Fallback console
    class Colors:
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        CYAN = '\033[96m'
        RESET = '\033[0m'
        BOLD = '\033[1m'
        DIM = '\033[2m'

    class Console:
        def info(self, msg, indent=0): print(f"{'  '*indent}[*] {msg}")
        def success(self, msg, indent=0): print(f"{'  '*indent}[+] {msg}")
        def warning(self, msg, indent=0): print(f"{'  '*indent}[!] {msg}")
        def error(self, msg, indent=0): print(f"{'  '*indent}[-] {msg}")
        def header(self, title, width=70):
            print(f"\n{Colors.CYAN}{'='*width}{Colors.RESET}")
            print(f"{Colors.BOLD}{title.center(width)}{Colors.RESET}")
            print(f"{Colors.CYAN}{'='*width}{Colors.RESET}")
        def subheader(self, title, width=70):
            print(f"\n{Colors.CYAN}{'-'*width}{Colors.RESET}")
            print(f"{title}")
            print(f"{Colors.CYAN}{'-'*width}{Colors.RESET}")
        def item(self, label, value, indent=1): print(f"{'  '*indent}{label}: {value}")
        def list_item(self, msg, indent=1): print(f"{'  '*indent}- {msg}")
        def progress_bar(self, current, total, width=30, label=""):
            filled = int(width * current / total)
            bar = "█" * filled + "░" * (width - filled)
            percent = current / total * 100
            print(f"\r[{bar}] {percent:5.1f}% {label}", end="", flush=True)
            if current >= total:
                print()
    console = Console()


# Legacy print functions for compatibility
def print_success(text): console.success(text)
def print_error(text): console.error(text)
def print_warning(text): console.warning(text)
def print_info(text): console.info(text)
def print_header(text): console.header(text)


# ============================================================================
# Utility Functions
# ============================================================================
def check_disk_space(required_gb: float, path: str = ".") -> bool:
    """Check if sufficient disk space is available"""
    try:
        import shutil
        total, used, free = shutil.disk_usage(path)
        free_gb = free / (1024 ** 3)

        if free_gb < required_gb:
            console.warning(f"Low disk space: {free_gb:.1f}GB free, {required_gb}GB required")
            return False
        return True
    except Exception:
        return True


def download_file(url: str, output_path: str, timeout: int = 300) -> bool:
    """Download file with progress bar"""
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))

        with open(output_path, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        console.progress_bar(downloaded, total_size, label=Path(output_path).name)

        return True
    except Exception as e:
        console.error(f"Download failed: {e}")
        return False


def extract_archive(archive_path: str, output_dir: str) -> bool:
    """Extract zip or tar archive"""
    try:
        console.info(f"Extracting to {output_dir}...")

        if archive_path.endswith('.zip'):
            with zipfile.ZipFile(archive_path, 'r') as zf:
                zf.extractall(output_dir)
        elif archive_path.endswith(('.tar.gz', '.tgz')):
            with tarfile.open(archive_path, 'r:gz') as tf:
                tf.extractall(output_dir)
        elif archive_path.endswith('.tar'):
            with tarfile.open(archive_path, 'r') as tf:
                tf.extractall(output_dir)
        else:
            console.error(f"Unknown archive format: {archive_path}")
            return False

        console.success("Extraction complete")
        return True
    except Exception as e:
        console.error(f"Extraction failed: {e}")
        return False


def verify_dataset(dataset_dir: str, min_wav_count: int = 100) -> Tuple[bool, str]:
    """Verify dataset was downloaded correctly"""
    if not os.path.exists(dataset_dir):
        return False, "Directory not found"

    wav_files = list(Path(dataset_dir).rglob('*.wav'))
    count = len(wav_files)

    if count < min_wav_count:
        return False, f"Only {count} .wav files found (expected >= {min_wav_count})"

    return True, f"{count} .wav files found"


# ============================================================================
# RAVDESS Downloader
# ============================================================================
def download_ravdess():
    """Download RAVDESS dataset from Zenodo"""
    console.header("DOWNLOADING RAVDESS")

    output_dir = f"{OUTPUT_BASE}/RAVDESS"
    os.makedirs(output_dir, exist_ok=True)

    # Check if already downloaded
    is_valid, msg = verify_dataset(output_dir, min_wav_count=1400)
    if is_valid:
        console.success(f"RAVDESS already downloaded! ({msg})")
        return True

    console.info("Dataset: Ryerson Audio-Visual Database of Emotional Speech")
    console.info("Size: ~1.5GB | Samples: ~1,440 files")
    console.info("Source: Zenodo")

    # URLs for each actor (split into smaller files)
    base_url = "https://zenodo.org/record/1188976/files"

    console.info("Downloading actor files...")

    success_count = 0
    for actor_num in range(1, 25):
        actor_file = f"Audio_Speech_Actors_{actor_num:02d}.zip"
        url = f"{base_url}/{actor_file}"
        zip_path = f"{TEMP_DIR}/{actor_file}"

        os.makedirs(TEMP_DIR, exist_ok=True)

        console.info(f"Actor {actor_num:02d}/24", indent=1)

        if download_file(url, zip_path, timeout=120):
            if extract_archive(zip_path, output_dir):
                success_count += 1
                os.remove(zip_path)

    # Verify
    is_valid, msg = verify_dataset(output_dir, min_wav_count=1400)
    if is_valid:
        console.success(f"RAVDESS downloaded successfully! ({msg})")
        return True
    else:
        console.error(f"Verification failed: {msg}")
        print_manual_instructions_ravdess()
        return False


def print_manual_instructions_ravdess():
    """Print manual download instructions for RAVDESS"""
    console.warning("MANUAL DOWNLOAD INSTRUCTIONS:")
    print("1. Visit: https://zenodo.org/record/1188976")
    print("2. Download: 'Audio_Speech_Actors_01-24.zip' (~1.5GB)")
    print(f"3. Extract to: {OUTPUT_BASE}/RAVDESS/")
    print("4. Run this script again to continue\n")


# ============================================================================
# TESS Downloader
# ============================================================================
def download_tess():
    """Download TESS with fallback methods"""
    console.header("DOWNLOADING TESS")

    output_dir = f"{OUTPUT_BASE}/TESS"
    os.makedirs(output_dir, exist_ok=True)

    # Check if already downloaded
    is_valid, msg = verify_dataset(output_dir, min_wav_count=2500)
    if is_valid:
        console.success(f"TESS already downloaded! ({msg})")
        return True

    console.info("Dataset: Toronto Emotional Speech Set")
    console.info("Size: ~400MB | Samples: ~2,800 files")

    # Method 1: Try Kaggle API
    console.info("[Method 1/3] Trying Kaggle API...")
    if download_tess_kaggle(output_dir):
        return True

    # Method 2: Try Hugging Face
    console.info("[Method 2/3] Trying Hugging Face...")
    if download_tess_huggingface(output_dir):
        return True

    # Method 3: Manual instructions
    console.warning("[Method 3/3] Automatic download failed")
    print_manual_instructions_tess()
    return False


def download_tess_kaggle(output_dir):
    """Try downloading TESS via Kaggle API"""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        console.info("Authenticating with Kaggle...")
        api = KaggleApi()
        api.authenticate()

        console.info("Downloading from Kaggle...")
        api.dataset_download_files(
            'ejlok1/toronto-emotional-speech-set-tess',
            path=output_dir,
            unzip=True
        )

        # Verify
        is_valid, msg = verify_dataset(output_dir, min_wav_count=2500)
        if is_valid:
            console.success(f"TESS downloaded via Kaggle! ({msg})")
            return True

        return False

    except ImportError:
        console.warning("Kaggle API not installed (pip install kaggle)")
        return False
    except Exception as e:
        console.warning(f"Kaggle download failed: {e}")
        return False


def download_tess_huggingface(output_dir):
    """Try downloading TESS via Hugging Face"""
    try:
        from datasets import load_dataset
        import soundfile as sf

        console.info("Loading from Hugging Face...")
        dataset = load_dataset("DynamicSuperb/EmotionRecognition_TESS", split="test")

        console.info(f"Processing {len(dataset)} samples...")

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

        console.success(f"TESS downloaded via Hugging Face! ({file_count} files)")
        return True

    except ImportError as e:
        console.warning(f"Missing library: {e}")
        console.info("Install: pip install datasets soundfile")
        return False
    except Exception as e:
        console.warning(f"Hugging Face download failed: {e}")
        return False


def print_manual_instructions_tess():
    """Print manual download instructions for TESS"""
    console.warning("MANUAL DOWNLOAD INSTRUCTIONS:")
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
    """Download CREMA-D from TalkBank"""
    console.header("DOWNLOADING CREMA-D")

    output_dir = f"{OUTPUT_BASE}/CREMA-D"
    os.makedirs(output_dir, exist_ok=True)

    # Check if already downloaded
    is_valid, msg = verify_dataset(output_dir, min_wav_count=7000)
    if is_valid:
        console.success(f"CREMA-D already downloaded! ({msg})")
        return True

    console.info("Dataset: Crowd Sourced Emotional Multimodal Actors Dataset")
    console.info("Size: ~2GB | Samples: ~7,442 files")
    console.info("Source: TalkBank (https://media.talkbank.org)")
    console.warning("Note: Large file, download may take 10-30 minutes")

    # Ask user confirmation
    response = input("\nProceed with download? [y/N]: ").strip().lower()
    if response != 'y':
        console.info("Skipping CREMA-D download")
        return False

    # URL
    url = "https://media.talkbank.org/ca/CREMA/AudioWAV.zip"
    zip_path = f"{TEMP_DIR}/crema_d.zip"

    # Download
    os.makedirs(TEMP_DIR, exist_ok=True)
    success = download_file(url, zip_path, timeout=600)  # 10min timeout

    if not success:
        console.error("Download failed!")
        print_manual_instructions_crema()
        return False

    # Extract
    success = extract_archive(zip_path, output_dir)

    if not success:
        console.error("Extraction failed!")
        return False

    # Verify
    is_valid, msg = verify_dataset(output_dir, min_wav_count=7000)
    if is_valid:
        console.success(f"CREMA-D downloaded successfully! ({msg})")
        return True
    else:
        console.error(f"Verification failed: {msg}")
        return False


def print_manual_instructions_crema():
    """Print manual download instructions for CREMA-D"""
    console.warning("MANUAL DOWNLOAD INSTRUCTIONS:")
    print("1. Visit: https://github.com/CheyneyComputerScience/CREMA-D")
    print("2. Or direct link: https://media.talkbank.org/ca/CREMA/AudioWAV.zip")
    print("3. Download AudioWAV.zip (~2GB)")
    print(f"4. Extract to: {OUTPUT_BASE}/CREMA-D/\n")


# ============================================================================
# Main Function
# ============================================================================
def main():
    """Main download orchestrator"""
    console.header("EVA PROJECT - DATASET DOWNLOADER")

    print("This script will download:")
    console.list_item("RAVDESS  (~1.5GB) - 1,440 samples")
    console.list_item("TESS     (~400MB) - 2,800 samples")
    console.list_item("CREMA-D  (~2GB)   - 7,442 samples")
    print(f"\nTotal: ~4GB, ~11,682 audio samples")
    print(f"Output directory: {OUTPUT_BASE}/")

    # Check disk space
    if not check_disk_space(5):
        response = input("\nContinue anyway? [y/N]: ").strip().lower()
        if response != 'y':
            console.info("Cancelled by user")
            return

    # Create directories
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    # Download each dataset
    results = {}

    console.info("Starting downloads...")

    results['RAVDESS'] = download_ravdess()
    results['TESS'] = download_tess()
    results['CREMA-D'] = download_crema_d()

    # Summary
    console.header("DOWNLOAD SUMMARY")

    for dataset, success in results.items():
        status = f"{Colors.GREEN}SUCCESS{Colors.RESET}" if success else f"{Colors.RED}FAILED{Colors.RESET}"
        print(f"  {dataset:12s}: [{status}]")

    successful = sum(results.values())
    total = len(results)

    print(f"\n  Total: {successful}/{total} datasets downloaded successfully")

    # Verification
    console.subheader("VERIFICATION")

    total_files = 0
    for dataset_name in ['RAVDESS', 'TESS', 'CREMA-D']:
        dataset_dir = f"{OUTPUT_BASE}/{dataset_name}"
        if os.path.exists(dataset_dir):
            wav_files = list(Path(dataset_dir).rglob('*.wav'))
            count = len(wav_files)
            total_files += count
            print(f"  {dataset_name:12s}: {count:5d} .wav files")
        else:
            print(f"  {dataset_name:12s}: {Colors.YELLOW}Not found{Colors.RESET}")

    print(f"\n  Total audio files: {total_files}")

    # Next steps
    if successful >= 2:
        console.header("READY FOR NEXT STEP")
        print("\nRun next:")
        print(f"   python Dataset/prepare_dataset.py")
    else:
        console.header("INCOMPLETE DOWNLOAD")
        print("\nPlease complete manual downloads for failed datasets")
        print("Then run this script again to verify")

    # Cleanup temp directory
    try:
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
            console.info("Cleaned up temporary files")
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
            console.header("VERIFYING DATASETS")
            for name in ['RAVDESS', 'TESS', 'CREMA-D']:
                dataset_dir = f"{OUTPUT_BASE}/{name}"
                is_valid, msg = verify_dataset(dataset_dir, min_wav_count=100)
                status = f"{Colors.GREEN}OK{Colors.RESET}" if is_valid else f"{Colors.RED}FAIL{Colors.RESET}"
                print(f"  {name:12s} [{status}] {msg}")
        elif command in ['--help', '-h']:
            print("Usage:")
            print("  python download_datasets.py              # Download all")
            print("  python download_datasets.py --ravdess    # RAVDESS only")
            print("  python download_datasets.py --tess       # TESS only")
            print("  python download_datasets.py --crema      # CREMA-D only")
            print("  python download_datasets.py --verify     # Verify downloads")
        else:
            console.error(f"Unknown command: {command}")
            print("Use --help for usage information")
    else:
        main()
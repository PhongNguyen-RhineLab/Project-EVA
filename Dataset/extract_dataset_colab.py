"""
Google Colab Dataset Extractor for EVA Project
Handles uploaded files in Colab environment
"""

import os
import zipfile
from pathlib import Path
from tqdm import tqdm
import shutil

# Configuration
OUTPUT_BASE = 'Dataset/prelabel_en'


class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")


def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")


def print_info(text):
    print(f"{Colors.BLUE}ℹ {text}{Colors.RESET}")


def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 70}{Colors.RESET}\n")


def detect_colab():
    """Check if running in Google Colab"""
    try:
        import google.colab
        return True
    except:
        return False


def find_zip_files_colab():
    """
    Find zip files in Colab environment
    Searches in:
    1. /content/ (Colab default upload location)
    2. /content/drive/MyDrive/ (if Google Drive mounted)
    3. Current directory and project directories
    """
    # Get current working directory
    cwd = os.getcwd()

    search_paths = [
        cwd,  # Current directory (highest priority)
        '/content',  # Colab default
        '/content/Project-EVA',  # Project folder
        '/content/EVA-Project',  # Alternative project name
        '/content/drive/MyDrive',  # Google Drive root
        '/content/drive/MyDrive/EVA',  # Project folder in Drive
        '/content/drive/MyDrive/Datasets',  # Datasets folder in Drive
        '.',  # Relative current directory
        'Dataset',  # Dataset folder
    ]

    found_files = {
        'ravdess': None,
        'tess': None,
        'crema': None
    }

    print_info("🔍 Searching for zip files in Colab environment...")
    print_info(f"Searching in: {', '.join(search_paths)}\n")

    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue

        print_info(f"Checking: {search_path}")

        try:
            files = os.listdir(search_path)
        except PermissionError:
            continue

        for file in files:
            file_lower = file.lower()
            file_path = os.path.join(search_path, file)

            # Skip if not a file or not a zip
            if not os.path.isfile(file_path) or not file_lower.endswith('.zip'):
                continue

            # Get file size for display
            size_mb = os.path.getsize(file_path) / (1024 * 1024)

            # Identify RAVDESS
            if 'ravdess' in file_lower or 'audio_speech_actors' in file_lower:
                if found_files['ravdess'] is None:
                    found_files['ravdess'] = file_path
                    print_success(f"Found RAVDESS: {file} ({size_mb:.1f} MB)")

            # Identify TESS
            elif 'tess' in file_lower or 'toronto' in file_lower:
                if found_files['tess'] is None:
                    found_files['tess'] = file_path
                    print_success(f"Found TESS: {file} ({size_mb:.1f} MB)")

            # Identify CREMA-D
            elif 'crema' in file_lower or 'audiowav' in file_lower:
                if found_files['crema'] is None:
                    found_files['crema'] = file_path
                    print_success(f"Found CREMA-D: {file} ({size_mb:.1f} MB)")

    return found_files


def list_all_files_in_content():
    """Debug function: List all files in /content"""
    print_header("📂 ALL FILES IN /CONTENT")

    if not os.path.exists('/content'):
        print_error("/content directory not found")
        return

    all_files = []
    for root, dirs, files in os.walk('/content'):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith('.zip'):
                size_mb = os.path.getsize(file_path) / (1024 * 1024)
                all_files.append((file_path, size_mb))

    if all_files:
        print_info("Found .zip files:")
        for path, size in all_files:
            print(f"  • {path} ({size:.1f} MB)")
    else:
        print_error("No .zip files found in /content")

    print()


def extract_zip(zip_path, output_dir, dataset_name, expected_count):
    """
    Generic zip extraction with progress bar
    """
    print_header(f"📦 EXTRACTING {dataset_name}")

    os.makedirs(output_dir, exist_ok=True)

    # Check if already extracted
    existing_wavs = list(Path(output_dir).rglob('*.wav'))
    if len(existing_wavs) >= expected_count * 0.95:
        print_success(f"{dataset_name} already extracted! ({len(existing_wavs)} files)")
        return True

    print_info(f"Source: {zip_path}")
    print_info(f"Destination: {output_dir}")
    print_info(f"Expected files: ~{expected_count}")

    try:
        # Check file size
        size_mb = os.path.getsize(zip_path) / (1024 * 1024)
        print_info(f"Zip file size: {size_mb:.1f} MB")

        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files
            members = zip_ref.namelist()
            wav_members = [m for m in members if m.lower().endswith('.wav')]

            print_info(f"Found {len(wav_members)} .wav files in archive")

            if len(wav_members) == 0:
                print_error("No .wav files found in zip!")
                return False

            # Extract with progress bar
            for member in tqdm(members, desc="Extracting", ncols=80):
                try:
                    zip_ref.extract(member, output_dir)
                except Exception as e:
                    # Skip problematic files
                    continue

        # Verify extraction
        extracted_wavs = list(Path(output_dir).rglob('*.wav'))
        actual_count = len(extracted_wavs)

        print_success(f"Extracted {actual_count} .wav files")

        if actual_count < expected_count * 0.8:
            print_error(f"Warning: Expected ~{expected_count}, got {actual_count}")
            return False

        return True

    except zipfile.BadZipFile:
        print_error("Corrupted zip file! Please re-upload.")
        return False
    except Exception as e:
        print_error(f"Extraction failed: {e}")
        return False


def organize_tess(output_dir):
    """Organize TESS by emotion"""
    print_info("Organizing TESS files by emotion...")

    emotion_map = {
        'angry': 'angry',
        'disgust': 'disgust',
        'fear': 'fearful',
        'happy': 'happy',
        'neutral': 'neutral',
        'ps': 'surprised',
        'sad': 'sad'
    }

    wav_files = list(Path(output_dir).rglob('*.wav'))
    organized = 0

    for wav_file in tqdm(wav_files, desc="Organizing", ncols=80):
        filename_lower = wav_file.stem.lower()

        detected_emotion = None
        for keyword, emotion in emotion_map.items():
            if keyword in filename_lower:
                detected_emotion = emotion
                break

        if detected_emotion:
            emotion_dir = os.path.join(output_dir, detected_emotion)
            os.makedirs(emotion_dir, exist_ok=True)

            if detected_emotion not in str(wav_file.parent).lower():
                new_path = os.path.join(emotion_dir, wav_file.name)
                shutil.move(str(wav_file), new_path)
                organized += 1

    if organized > 0:
        print_success(f"Organized {organized} files")


def flatten_crema(output_dir):
    """Flatten CREMA-D structure"""
    print_info("Flattening CREMA-D structure...")

    wav_files = list(Path(output_dir).rglob('*.wav'))
    moved = 0

    for wav_file in tqdm(wav_files, desc="Flattening", ncols=80):
        if wav_file.parent != Path(output_dir):
            new_path = os.path.join(output_dir, wav_file.name)

            if os.path.exists(new_path):
                base, ext = os.path.splitext(wav_file.name)
                counter = 1
                while os.path.exists(new_path):
                    new_path = os.path.join(output_dir, f"{base}_{counter}{ext}")
                    counter += 1

            shutil.move(str(wav_file), new_path)
            moved += 1

    if moved > 0:
        print_success(f"Moved {moved} files")


def verify_all():
    """Verify all datasets"""
    print_header("📁 DATASET VERIFICATION")

    datasets = {
        'RAVDESS': {'path': f'{OUTPUT_BASE}/RAVDESS', 'expected': 1440},
        'TESS': {'path': f'{OUTPUT_BASE}/TESS', 'expected': 2800},
        'CREMA-D': {'path': f'{OUTPUT_BASE}/CREMA-D', 'expected': 7442}
    }

    total_files = 0
    results = []

    for name, info in datasets.items():
        if os.path.exists(info['path']):
            wav_files = list(Path(info['path']).rglob('*.wav'))
            count = len(wav_files)
            total_files += count
            percentage = (count / info['expected']) * 100

            if count >= info['expected'] * 0.95:
                print_success(f"{name:12s}: {count:5d} files ({percentage:.0f}%)")
                results.append(True)
            elif count > 0:
                print_info(f"{name:12s}: {count:5d} files ({percentage:.0f}%) - Incomplete")
                results.append(False)
            else:
                print_error(f"{name:12s}: Not found")
                results.append(False)
        else:
            print_error(f"{name:12s}: Directory not found")
            results.append(False)

    print(f"\n{'=' * 70}")
    print(f"Total WAV files: {total_files}")
    print(f"Expected total: ~11,682")
    print(f"Completion: {(total_files / 11682) * 100:.1f}%")
    print(f"{'=' * 70}\n")

    return all(results)


def upload_files_ui():
    """Colab file upload UI"""
    if not detect_colab():
        print_error("Not running in Google Colab")
        return

    from google.colab import files

    print_header("📤 UPLOAD ZIP FILES")
    print("Please upload your dataset zip files:")
    print("  • RAVDESS (Audio_Speech_Actors_01-24.zip) - ~1.5GB")
    print("  • TESS (tess.zip) - ~400MB")
    print("  • CREMA-D (AudioWAV.zip) - ~2GB")
    print("\nNote: Large files may take time to upload\n")

    uploaded = files.upload()

    print_success(f"Uploaded {len(uploaded)} file(s)")
    for filename in uploaded.keys():
        size_mb = len(uploaded[filename]) / (1024 * 1024)
        print(f"  • {filename} ({size_mb:.1f} MB)")


def mount_google_drive():
    """Mount Google Drive in Colab"""
    if not detect_colab():
        return False

    from google.colab import drive

    print_header("💾 MOUNTING GOOGLE DRIVE")
    print("This allows access to files stored in your Google Drive\n")

    try:
        drive.mount('/content/drive')
        print_success("Google Drive mounted at /content/drive")
        return True
    except Exception as e:
        print_error(f"Failed to mount Drive: {e}")
        return False


def main():
    """Main workflow for Colab"""
    print_header("🎯 EVA PROJECT - COLAB DATASET EXTRACTOR")

    if not detect_colab():
        print_error("This script is designed for Google Colab!")
        print_info("Use 'extract_manual_datasets.py' for local environment")
        return

    print("Running in Google Colab ✓\n")

    # Option 1: Check if files already uploaded
    print_info("Option 1: Searching for uploaded files...")
    found = find_zip_files_colab()

    found_count = sum(1 for v in found.values() if v is not None)

    if found_count == 0:
        print_error("\nNo zip files found!")
        print("\nYou have 3 options:")
        print("  1. Upload files now (run: upload_files_ui())")
        print("  2. Mount Google Drive (run: mount_google_drive())")
        print("  3. Upload via Colab UI (Files → Upload)")

        # Debug: Show all files
        list_all_files_in_content()
        return

    print_success(f"\nFound {found_count}/3 datasets\n")

    # Extract datasets
    if found['ravdess']:
        success = extract_zip(
            found['ravdess'],
            f"{OUTPUT_BASE}/RAVDESS",
            "RAVDESS",
            1440
        )

    if found['tess']:
        success = extract_zip(
            found['tess'],
            f"{OUTPUT_BASE}/TESS",
            "TESS",
            2800
        )
        if success:
            organize_tess(f"{OUTPUT_BASE}/TESS")

    if found['crema']:
        success = extract_zip(
            found['crema'],
            f"{OUTPUT_BASE}/CREMA-D",
            "CREMA-D",
            7442
        )
        if success:
            flatten_crema(f"{OUTPUT_BASE}/CREMA-D")

    # Verify
    all_ok = verify_all()

    if all_ok:
        print_header("✅ ALL DATASETS READY!")
        print("\n🚀 Next step:")
        print("   !python Dataset/prepare_dataset.py")
    else:
        print_header("⚠ INCOMPLETE SETUP")
        print("\nUpload missing datasets and run again")


# Convenience functions for Colab notebook
def quick_setup():
    """Quick setup: mount drive and extract"""
    if detect_colab():
        mount_google_drive()
    main()


def debug_files():
    """Debug: Show all files in /content"""
    list_all_files_in_content()
    print("\nSearching for datasets...")
    found = find_zip_files_colab()
    print(f"\nFound datasets: {found}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()

        if cmd == '--upload':
            upload_files_ui()
        elif cmd == '--mount':
            mount_google_drive()
        elif cmd == '--verify':
            verify_all()
        elif cmd == '--debug':
            debug_files()
        elif cmd == '--help':
            print("Colab Dataset Extractor Commands:")
            print("  python colab_extract_datasets.py              # Run main extraction")
            print("  python colab_extract_datasets.py --upload     # Upload files UI")
            print("  python colab_extract_datasets.py --mount      # Mount Google Drive")
            print("  python colab_extract_datasets.py --verify     # Verify datasets")
            print("  python colab_extract_datasets.py --debug      # Debug file locations")
    else:
        main()
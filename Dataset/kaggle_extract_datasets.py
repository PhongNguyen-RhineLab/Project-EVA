"""
Kaggle Notebook Dataset Extractor for EVA Project
Optimized for Kaggle environment with dataset mounting
"""

import os
import zipfile
from pathlib import Path
from tqdm import tqdm
import shutil
import sys

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


def detect_kaggle():
    """Check if running in Kaggle"""
    return os.path.exists('/kaggle')


def find_zip_files_kaggle():
    """
    Find zip files in Kaggle environment
    Searches in:
    1. /kaggle/input/ (Kaggle datasets)
    2. /kaggle/working/ (uploaded files)
    3. Current directory
    """
    search_paths = [
        '/kaggle/input',  # Kaggle datasets (highest priority)
        '/kaggle/working',  # Uploaded files
        '.',  # Current directory
        'Dataset',  # Dataset folder
    ]

    found_files = {
        'ravdess': None,
        'tess': None,
        'crema': None
    }

    print_info("🔍 Searching for zip files in Kaggle environment...")

    for search_path in search_paths:
        if not os.path.exists(search_path):
            continue

        print_info(f"Checking: {search_path}")

        # Recursively search in subdirectories (Kaggle datasets structure)
        for root, dirs, files in os.walk(search_path):
            for file in files:
                file_lower = file.lower()
                file_path = os.path.join(root, file)

                # Skip if not a zip file
                if not file_lower.endswith('.zip'):
                    continue

                # Get file size for display
                try:
                    size_mb = os.path.getsize(file_path) / (1024 * 1024)
                except:
                    size_mb = 0

                # Identify RAVDESS
                if ('ravdess' in file_lower or 'audio_speech_actors' in file_lower):
                    if found_files['ravdess'] is None:
                        found_files['ravdess'] = file_path
                        print_success(f"Found RAVDESS: {file} ({size_mb:.1f} MB)")

                # Identify TESS
                elif ('tess' in file_lower or 'toronto' in file_lower):
                    if found_files['tess'] is None:
                        found_files['tess'] = file_path
                        print_success(f"Found TESS: {file} ({size_mb:.1f} MB)")

                # Identify CREMA-D
                elif ('crema' in file_lower or 'audiowav' in file_lower):
                    if found_files['crema'] is None:
                        found_files['crema'] = file_path
                        print_success(f"Found CREMA-D: {file} ({size_mb:.1f} MB)")

    return found_files


def list_kaggle_datasets():
    """List all available Kaggle datasets"""
    print_header("📂 KAGGLE INPUT DATASETS")

    input_path = '/kaggle/input'
    if not os.path.exists(input_path):
        print_error("No Kaggle datasets found")
        return

    datasets = os.listdir(input_path)

    if not datasets:
        print_error("No datasets mounted")
        print_info("\nTo add datasets:")
        print("  1. Click 'Add Data' in notebook")
        print("  2. Search for: ravdess, tess, crema-d")
        print("  3. Add to notebook")
        return

    print_info(f"Found {len(datasets)} dataset(s):")
    for dataset in datasets:
        dataset_path = os.path.join(input_path, dataset)

        # Count files
        total_files = 0
        zip_files = []
        for root, dirs, files in os.walk(dataset_path):
            total_files += len(files)
            zip_files.extend([f for f in files if f.endswith('.zip')])

        print(f"\n  📁 {dataset}")
        print(f"     Path: {dataset_path}")
        print(f"     Files: {total_files}")
        if zip_files:
            print(f"     Zip files: {', '.join(zip_files)}")


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
                try:
                    shutil.move(str(wav_file), new_path)
                    organized += 1
                except:
                    pass

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

            try:
                shutil.move(str(wav_file), new_path)
                moved += 1
            except:
                pass

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


def check_kaggle_disk_space():
    """Check available disk space in Kaggle"""
    import shutil

    stat = shutil.disk_usage('/kaggle/working')

    total_gb = stat.total / (1024 ** 3)
    used_gb = stat.used / (1024 ** 3)
    free_gb = stat.free / (1024 ** 3)

    print_info(f"Kaggle Disk Space:")
    print(f"  Total: {total_gb:.1f} GB")
    print(f"  Used:  {used_gb:.1f} GB")
    print(f"  Free:  {free_gb:.1f} GB")

    if free_gb < 3:
        print_error("Low disk space! May not fit all datasets.")
        return False

    return True


def main():
    """Main extraction workflow for Kaggle"""
    print_header("🎯 EVA PROJECT - KAGGLE DATASET EXTRACTOR")

    if not detect_kaggle():
        print_error("This script is designed for Kaggle Notebooks!")
        print_info("Use 'colab_extract_datasets.py' for Google Colab")
        print_info("Use 'extract_manual_datasets.py' for local environment")
        return

    print("Running in Kaggle Notebook ✓\n")

    # Check disk space
    check_kaggle_disk_space()

    # List available datasets
    list_kaggle_datasets()

    # Find zip files
    print("\n")
    found = find_zip_files_kaggle()

    found_count = sum(1 for v in found.values() if v is not None)

    if found_count == 0:
        print_error("\nNo zip files found!")
        print("\n" + "=" * 70)
        print("SETUP INSTRUCTIONS FOR KAGGLE:")
        print("=" * 70)
        print("\n1. Add Datasets to Notebook:")
        print("   - Click 'Add Data' button (right sidebar)")
        print("   - Search for datasets:")
        print("     • 'ravdess audio speech'")
        print("     • 'tess toronto emotional speech'")
        print("     • 'crema-d emotional'")
        print("   - Add them to your notebook")
        print("\n2. Or Upload Manually:")
        print("   - Click 'Add Data' → 'Upload'")
        print("   - Select your 3 zip files")
        print("   - Wait for upload to complete")
        print("\n3. Re-run this script")
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
        print("\nAdd missing datasets and run again")


# Convenience functions for Kaggle notebook
def quick_setup():
    """Quick setup for Kaggle"""
    main()


def debug_files():
    """Debug: Show all files in Kaggle environment"""
    print_header("📂 KAGGLE ENVIRONMENT STRUCTURE")

    print("\n1. Kaggle Input Datasets:")
    if os.path.exists('/kaggle/input'):
        for item in os.listdir('/kaggle/input'):
            print(f"   • /kaggle/input/{item}")
    else:
        print("   (none)")

    print("\n2. Kaggle Working Directory:")
    if os.path.exists('/kaggle/working'):
        items = os.listdir('/kaggle/working')
        if items:
            for item in items[:10]:  # First 10
                print(f"   • /kaggle/working/{item}")
        else:
            print("   (empty)")

    print("\n3. Searching for .zip files...")
    found = find_zip_files_kaggle()
    print(f"\n   Found: {found}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()

        if cmd == '--list':
            list_kaggle_datasets()
        elif cmd == '--verify':
            verify_all()
        elif cmd == '--debug':
            debug_files()
        elif cmd == '--help':
            print("Kaggle Dataset Extractor Commands:")
            print("  python kaggle_extract_datasets.py              # Run main extraction")
            print("  python kaggle_extract_datasets.py --list       # List Kaggle datasets")
            print("  python kaggle_extract_datasets.py --verify     # Verify datasets")
            print("  python kaggle_extract_datasets.py --debug      # Debug file locations")
    else:
        main()
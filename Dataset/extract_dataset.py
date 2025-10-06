"""
Extract and Organize Manually Downloaded Datasets
For TESS and CREMA-D zip files already in project
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
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.RESET}\n")


def find_zip_files(search_dirs=None):
    """
    Find RAVDESS, TESS and CREMA-D zip files in project
    """
    if search_dirs is None:
        # Search in common locations
        search_dirs = [
            '.',                    # Project root
            'Dataset',              # Dataset folder
            'Dataset/manual',       # Manual downloads
            'downloads',            # Downloads folder
            os.path.expanduser('~/Downloads'),  # User downloads
        ]

    found_files = {
        'ravdess': None,
        'tess': None,
        'crema': None
    }

    print_info("Searching for zip files...")

    for directory in search_dirs:
        if not os.path.exists(directory):
            continue

        for file in os.listdir(directory):
            file_lower = file.lower()
            file_path = os.path.join(directory, file)

            # Check if it's a zip file
            if not file_lower.endswith('.zip'):
                continue

            # Identify RAVDESS
            if 'ravdess' in file_lower and found_files['ravdess'] is None:
                found_files['ravdess'] = file_path
                print_success(f"Found RAVDESS: {file_path}")

            # Identify TESS
            elif 'tess' in file_lower and found_files['tess'] is None:
                found_files['tess'] = file_path
                print_success(f"Found TESS: {file_path}")

            # Identify CREMA-D
            elif ('crema' in file_lower or 'audiowav' in file_lower) and found_files['crema'] is None:
                found_files['crema'] = file_path
                print_success(f"Found CREMA-D: {file_path}")

    return found_files


def extract_tess(zip_path, output_dir):
    """
    Extract TESS dataset
    """
    print_header("📦 EXTRACTING TESS")

    os.makedirs(output_dir, exist_ok=True)

    # Check if already extracted
    existing_wavs = list(Path(output_dir).rglob('*.wav'))
    if len(existing_wavs) > 2500:
        print_success(f"TESS already extracted! ({len(existing_wavs)} files)")
        return True

    print_info(f"Source: {zip_path}")
    print_info(f"Destination: {output_dir}")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files
            members = zip_ref.namelist()
            wav_members = [m for m in members if m.lower().endswith('.wav')]

            print_info(f"Found {len(wav_members)} .wav files in archive")

            # Extract with progress bar
            for member in tqdm(members, desc="Extracting"):
                try:
                    zip_ref.extract(member, output_dir)
                except Exception as e:
                    # Skip problematic files
                    continue

        # Verify extraction
        extracted_wavs = list(Path(output_dir).rglob('*.wav'))
        print_success(f"Extracted {len(extracted_wavs)} .wav files")

        # Organize TESS structure if needed
        organize_tess_structure(output_dir)

        return True

    except Exception as e:
        print_error(f"Extraction failed: {e}")
        return False


def organize_tess_structure(output_dir):
    """
    Organize TESS files by emotion if not already organized
    """
    print_info("Checking TESS structure...")

    # TESS emotions mapping
    emotion_keywords = {
        'angry': 'angry',
        'disgust': 'disgust',
        'fear': 'fearful',
        'happy': 'happy',
        'neutral': 'neutral',
        'ps': 'surprised',  # pleasant surprise
        'sad': 'sad'
    }

    # Find all wav files
    wav_files = list(Path(output_dir).rglob('*.wav'))

    # Check if already organized (files in emotion subdirectories)
    organized_count = sum(1 for f in wav_files if any(emotion in str(f.parent).lower() for emotion in emotion_keywords.values()))

    if organized_count > len(wav_files) * 0.8:  # 80% already organized
        print_success("TESS already organized by emotion")
        return

    print_info("Organizing TESS files by emotion...")

    organized = 0
    for wav_file in tqdm(wav_files, desc="Organizing"):
        filename_lower = wav_file.stem.lower()

        # Detect emotion from filename
        detected_emotion = None
        for keyword, emotion in emotion_keywords.items():
            if keyword in filename_lower:
                detected_emotion = emotion
                break

        if detected_emotion:
            # Create emotion directory
            emotion_dir = os.path.join(output_dir, detected_emotion)
            os.makedirs(emotion_dir, exist_ok=True)

            # Move file if not already in emotion directory
            if detected_emotion not in str(wav_file.parent).lower():
                new_path = os.path.join(emotion_dir, wav_file.name)
                shutil.move(str(wav_file), new_path)
                organized += 1

    if organized > 0:
        print_success(f"Organized {organized} files into emotion directories")


def extract_crema(zip_path, output_dir):
    """
    Extract CREMA-D dataset
    """
    print_header("📦 EXTRACTING CREMA-D")

    os.makedirs(output_dir, exist_ok=True)

    # Check if already extracted
    existing_wavs = list(Path(output_dir).rglob('*.wav'))
    if len(existing_wavs) > 7000:
        print_success(f"CREMA-D already extracted! ({len(existing_wavs)} files)")
        return True

    print_info(f"Source: {zip_path}")
    print_info(f"Destination: {output_dir}")
    print_info("Note: This is a large file (~2GB), extraction may take a few minutes")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Get list of files
            members = zip_ref.namelist()
            wav_members = [m for m in members if m.lower().endswith('.wav')]

            print_info(f"Found {len(wav_members)} .wav files in archive")

            # Extract with progress bar
            for member in tqdm(members, desc="Extracting"):
                try:
                    zip_ref.extract(member, output_dir)
                except Exception as e:
                    continue

        # Verify extraction
        extracted_wavs = list(Path(output_dir).rglob('*.wav'))
        print_success(f"Extracted {len(extracted_wavs)} .wav files")

        # Flatten structure if needed (CREMA-D sometimes has nested folders)
        flatten_crema_structure(output_dir)

        return True

    except Exception as e:
        print_error(f"Extraction failed: {e}")
        return False


def flatten_crema_structure(output_dir):
    """
    Flatten CREMA-D directory structure (all wav files in one folder)
    """
    print_info("Flattening CREMA-D structure...")

    # Find all wav files
    wav_files = list(Path(output_dir).rglob('*.wav'))

    # Check if files are already at root level
    root_wavs = [f for f in wav_files if f.parent == Path(output_dir)]

    if len(root_wavs) > len(wav_files) * 0.9:  # 90% at root
        print_success("CREMA-D structure already flat")
        return

    print_info("Moving files to root directory...")

    moved = 0
    for wav_file in tqdm(wav_files, desc="Flattening"):
        if wav_file.parent != Path(output_dir):
            new_path = os.path.join(output_dir, wav_file.name)

            # Handle duplicates
            if os.path.exists(new_path):
                base, ext = os.path.splitext(wav_file.name)
                counter = 1
                while os.path.exists(new_path):
                    new_path = os.path.join(output_dir, f"{base}_{counter}{ext}")
                    counter += 1

            shutil.move(str(wav_file), new_path)
            moved += 1

    if moved > 0:
        print_success(f"Moved {moved} files to root directory")

        # Clean up empty subdirectories
        for dirpath, dirnames, filenames in os.walk(output_dir, topdown=False):
            if dirpath != output_dir and not filenames and not dirnames:
                try:
                    os.rmdir(dirpath)
                except:
                    pass


def verify_all_datasets():
    """
    Verify all three datasets
    """
    print_header("📁 DATASET VERIFICATION")

    datasets = {
        'RAVDESS': {'path': f'{OUTPUT_BASE}/RAVDESS', 'expected': 1440},
        'TESS': {'path': f'{OUTPUT_BASE}/TESS', 'expected': 2800},
        'CREMA-D': {'path': f'{OUTPUT_BASE}/CREMA-D', 'expected': 7442}
    }

    total_files = 0
    all_ok = True

    for name, info in datasets.items():
        if os.path.exists(info['path']):
            wav_files = list(Path(info['path']).rglob('*.wav'))
            count = len(wav_files)
            total_files += count

            percentage = (count / info['expected']) * 100 if info['expected'] > 0 else 0

            if count >= info['expected'] * 0.95:  # 95% threshold
                print_success(f"{name:12s}: {count:5d} files ({percentage:.0f}%)")
            elif count > 0:
                print_info(f"{name:12s}: {count:5d} files ({percentage:.0f}%) - Incomplete")
                all_ok = False
            else:
                print_error(f"{name:12s}: Not found")
                all_ok = False
        else:
            print_error(f"{name:12s}: Directory not found")
            all_ok = False

    print(f"\n{'='*70}")
    print(f"Total WAV files: {total_files}")
    print(f"Expected total: ~11,682")
    print(f"{'='*70}")

    return all_ok


def main():
    """
    Main extraction workflow
    """
    print_header("🎯 EVA PROJECT - MANUAL DATASET EXTRACTOR")

    print("This script will:")
    print("  1. Find TESS and CREMA-D zip files in your project")
    print("  2. Extract them to the correct locations")
    print("  3. Organize the file structure")
    print("  4. Verify all datasets\n")

    # Find zip files
    found_files = find_zip_files()

    # Check what we found
    if found_files['tess'] is None and found_files['crema'] is None:
        print_error("\nNo zip files found!")
        print_info("\nPlease place your zip files in one of these locations:")
        print("  - Project root directory")
        print("  - Dataset/ folder")
        print("  - Dataset/manual/ folder")
        print("\nExpected filenames:")
        print("  - TESS: *tess*.zip")
        print("  - CREMA-D: *crema*.zip or *AudioWAV*.zip")
        return

    # Extract TESS
    if found_files['tess']:
        tess_output = f"{OUTPUT_BASE}/TESS"
        success = extract_tess(found_files['tess'], tess_output)
        if not success:
            print_error("TESS extraction failed!")
    else:
        print_info("\nTESS zip not found - skipping")

    # Extract CREMA-D
    if found_files['crema']:
        crema_output = f"{OUTPUT_BASE}/CREMA-D"
        success = extract_crema(found_files['crema'], crema_output)
        if not success:
            print_error("CREMA-D extraction failed!")
    else:
        print_info("\nCREMA-D zip not found - skipping")

    # Verify all datasets
    all_ok = verify_all_datasets()

    # Next steps
    if all_ok:
        print_header("✅ ALL DATASETS READY!")
        print("\n🚀 Next step:")
        print("   python Dataset/prepare_dataset.py")
    else:
        print_header("⚠ INCOMPLETE SETUP")
        print("\nSome datasets are missing or incomplete.")
        print("Please check the messages above and fix any issues.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == '--tess':
            # Extract TESS only
            found = find_zip_files()
            if found['tess']:
                extract_tess(found['tess'], f"{OUTPUT_BASE}/TESS")
            else:
                print_error("TESS zip file not found")

        elif command == '--crema':
            # Extract CREMA-D only
            found = find_zip_files()
            if found['crema']:
                extract_crema(found['crema'], f"{OUTPUT_BASE}/CREMA-D")
            else:
                print_error("CREMA-D zip file not found")

        elif command == '--verify':
            # Verify only
            verify_all_datasets()

        elif command in ['--help', '-h']:
            print("Usage:")
            print("  python extract_manual_datasets.py           # Extract all")
            print("  python extract_manual_datasets.py --tess    # Extract TESS only")
            print("  python extract_manual_datasets.py --crema   # Extract CREMA-D only")
            print("  python extract_manual_datasets.py --verify  # Verify datasets")
        else:
            print_error(f"Unknown command: {command}")
            print("Use --help for usage")
    else:
        main()
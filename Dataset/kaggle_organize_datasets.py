"""
Kaggle Dataset Organizer for EVA Project
Since Kaggle auto-extracts zips, this script only organizes files
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Configuration
INPUT_BASE = '/kaggle/input'
OUTPUT_BASE = '/kaggle/working/Dataset/prelabel_en'

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


def list_kaggle_datasets():
    """List all datasets in /kaggle/input/"""
    print_header("📂 KAGGLE INPUT DATASETS")

    if not os.path.exists(INPUT_BASE):
        print_error("Kaggle input directory not found!")
        return []

    datasets = os.listdir(INPUT_BASE)

    if not datasets:
        print_error("No datasets found!")
        print_info("\nTo add datasets:")
        print("  1. Click 'Add Data' (right sidebar)")
        print("  2. Search: ravdess, tess, crema-d")
        print("  3. Add to notebook")
        return []

    print_info(f"Found {len(datasets)} dataset(s):\n")

    dataset_info = []
    for dataset in sorted(datasets):
        dataset_path = os.path.join(INPUT_BASE, dataset)

        # Count .wav files
        wav_files = list(Path(dataset_path).rglob('*.wav'))
        wav_count = len(wav_files)

        # Get size
        try:
            size_mb = sum(f.stat().st_size for f in Path(dataset_path).rglob('*') if f.is_file()) / (1024**2)
        except:
            size_mb = 0

        print(f"  📁 {dataset}")
        print(f"     Path: {dataset_path}")
        print(f"     WAV files: {wav_count}")
        print(f"     Size: {size_mb:.1f} MB\n")

        dataset_info.append({
            'name': dataset,
            'path': dataset_path,
            'wav_count': wav_count
        })

    return dataset_info


def find_dataset_folders():
    """Find RAVDESS, TESS, CREMA-D folders in Kaggle input"""
    print_header("🔍 LOCATING DATASETS")

    datasets = list_kaggle_datasets()

    found = {
        'ravdess': None,
        'tess': None,
        'crema': None
    }

    for dataset in datasets:
        name_lower = dataset['name'].lower()

        if 'ravdess' in name_lower or 'ryerson' in name_lower:
            found['ravdess'] = dataset['path']
            print_success(f"RAVDESS: {dataset['path']} ({dataset['wav_count']} files)")

        elif 'tess' in name_lower or 'toronto' in name_lower:
            found['tess'] = dataset['path']
            print_success(f"TESS: {dataset['path']} ({dataset['wav_count']} files)")

        elif 'crema' in name_lower:
            found['crema'] = dataset['path']
            print_success(f"CREMA-D: {dataset['path']} ({dataset['wav_count']} files)")

    # Count found
    found_count = sum(1 for v in found.values() if v is not None)

    if found_count == 0:
        print_error("\nNo emotion datasets found!")
        print_info("\nExpected dataset names containing:")
        print("  • 'ravdess' or 'ryerson'")
        print("  • 'tess' or 'toronto'")
        print("  • 'crema'")
    else:
        print_success(f"\n✓ Found {found_count}/3 datasets")

    return found


def copy_dataset(source_path, dest_path, dataset_name):
    """Copy dataset from input to working directory"""
    print_header(f"📋 COPYING {dataset_name}")

    os.makedirs(dest_path, exist_ok=True)

    # Find all .wav files
    wav_files = list(Path(source_path).rglob('*.wav'))

    if not wav_files:
        print_error(f"No .wav files found in {source_path}")
        return False

    print_info(f"Source: {source_path}")
    print_info(f"Destination: {dest_path}")
    print_info(f"Files to copy: {len(wav_files)}\n")

    # Copy with progress bar
    copied = 0
    for wav_file in tqdm(wav_files, desc="Copying", ncols=80):
        try:
            # Preserve directory structure or flatten?
            # For now, flatten to avoid nested folders
            dest_file = os.path.join(dest_path, wav_file.name)

            # Handle duplicates
            if os.path.exists(dest_file):
                base, ext = os.path.splitext(wav_file.name)
                counter = 1
                while os.path.exists(dest_file):
                    dest_file = os.path.join(dest_path, f"{base}_{counter}{ext}")
                    counter += 1

            shutil.copy2(str(wav_file), dest_file)
            copied += 1
        except Exception as e:
            print_error(f"Failed to copy {wav_file.name}: {e}")
            continue

    print_success(f"Copied {copied}/{len(wav_files)} files")
    return copied > 0


def organize_tess(tess_path):
    """Organize TESS files by emotion if needed"""
    print_info("Checking TESS organization...")

    emotion_map = {
        'angry': 'angry',
        'disgust': 'disgust',
        'fear': 'fearful',
        'happy': 'happy',
        'neutral': 'neutral',
        'ps': 'surprised',
        'sad': 'sad'
    }

    wav_files = list(Path(tess_path).rglob('*.wav'))

    # Check if already organized
    organized = sum(1 for f in wav_files if any(emo in str(f.parent).lower() for emo in emotion_map.values()))

    if organized > len(wav_files) * 0.8:
        print_success("TESS already organized by emotion")
        return

    print_info("Organizing TESS files by emotion...")

    organized_count = 0
    for wav_file in tqdm(wav_files, desc="Organizing", ncols=80):
        filename_lower = wav_file.stem.lower()

        detected_emotion = None
        for keyword, emotion in emotion_map.items():
            if keyword in filename_lower:
                detected_emotion = emotion
                break

        if detected_emotion:
            emotion_dir = os.path.join(tess_path, detected_emotion)
            os.makedirs(emotion_dir, exist_ok=True)

            if detected_emotion not in str(wav_file.parent).lower():
                new_path = os.path.join(emotion_dir, wav_file.name)
                try:
                    shutil.move(str(wav_file), new_path)
                    organized_count += 1
                except:
                    pass

    if organized_count > 0:
        print_success(f"Organized {organized_count} files into emotion folders")


def flatten_directory(directory):
    """Move all .wav files to root of directory"""
    print_info(f"Flattening {directory}...")

    wav_files = list(Path(directory).rglob('*.wav'))
    root_files = [f for f in wav_files if f.parent == Path(directory)]

    if len(root_files) > len(wav_files) * 0.9:
        print_success("Directory already flat")
        return

    moved = 0
    for wav_file in tqdm(wav_files, desc="Flattening", ncols=80):
        if wav_file.parent != Path(directory):
            new_path = os.path.join(directory, wav_file.name)

            # Handle duplicates
            if os.path.exists(new_path):
                base, ext = os.path.splitext(wav_file.name)
                counter = 1
                while os.path.exists(new_path):
                    new_path = os.path.join(directory, f"{base}_{counter}{ext}")
                    counter += 1

            try:
                shutil.move(str(wav_file), new_path)
                moved += 1
            except:
                pass

    if moved > 0:
        print_success(f"Moved {moved} files to root directory")


def verify_datasets():
    """Verify all datasets are ready"""
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

    print(f"\n{'='*70}")
    print(f"Total WAV files: {total_files}")
    print(f"Expected total: ~11,682")
    print(f"Completion: {(total_files/11682)*100:.1f}%")
    print(f"{'='*70}\n")

    return all(results)


def main():
    """Main workflow for Kaggle"""
    print_header("🎯 EVA PROJECT - KAGGLE DATASET ORGANIZER")

    print("Since Kaggle auto-extracts datasets, this script:")
    print("  1. Locates extracted datasets in /kaggle/input/")
    print("  2. Copies them to /kaggle/working/")
    print("  3. Organizes file structure")
    print("  4. Verifies completeness\n")

    # Check Kaggle environment
    if not os.path.exists('/kaggle'):
        print_error("Not running in Kaggle environment!")
        print_info("This script is for Kaggle Notebooks only")
        return

    print_success("Running in Kaggle ✓\n")

    # Find datasets
    found = find_dataset_folders()

    found_count = sum(1 for v in found.values() if v is not None)

    if found_count == 0:
        print_error("\nNo datasets found!")
        print("\n" + "="*70)
        print("SETUP INSTRUCTIONS:")
        print("="*70)
        print("\n1. Click 'Add Data' button (right sidebar)")
        print("2. Search for datasets:")
        print("   • Search: 'ravdess emotional speech'")
        print("   • Search: 'tess toronto emotional'")
        print("   • Search: 'crema-d emotional'")
        print("3. Click 'Add' for each dataset")
        print("4. Wait for datasets to mount")
        print("5. Re-run this script\n")
        return

    print_info(f"\nProceeding with {found_count}/3 datasets...")

    # Copy RAVDESS
    if found['ravdess']:
        copy_dataset(
            found['ravdess'],
            f"{OUTPUT_BASE}/RAVDESS",
            "RAVDESS"
        )

    # Copy TESS
    if found['tess']:
        success = copy_dataset(
            found['tess'],
            f"{OUTPUT_BASE}/TESS",
            "TESS"
        )
        if success:
            organize_tess(f"{OUTPUT_BASE}/TESS")

    # Copy CREMA-D
    if found['crema']:
        success = copy_dataset(
            found['crema'],
            f"{OUTPUT_BASE}/CREMA-D",
            "CREMA-D"
        )
        if success:
            flatten_directory(f"{OUTPUT_BASE}/CREMA-D")

    # Verify
    all_ok = verify_datasets()

    if all_ok:
        print_header("✅ ALL DATASETS READY!")
        print("\n🚀 Next step:")
        print("   !python Dataset/prepare_dataset.py")
        print("\n💡 Note: Dataset files are in /kaggle/working/")
        print("   They will persist for this session only")
    else:
        print_header("⚠ SETUP INCOMPLETE")
        print("\nSome datasets are missing or incomplete")
        print("Check the messages above for details")


def quick_check():
    """Quick check of available datasets"""
    datasets = list_kaggle_datasets()

    print("\n" + "="*70)
    print("QUICK SUMMARY")
    print("="*70)

    total_wavs = sum(d['wav_count'] for d in datasets)
    print(f"Total datasets: {len(datasets)}")
    print(f"Total .wav files: {total_wavs}")

    if total_wavs > 10000:
        print_success("\n✓ Sufficient audio files for training!")
    elif total_wavs > 5000:
        print_info("\n⚠ Moderate amount of files. Consider adding more datasets.")
    else:
        print_error("\n✗ Insufficient files. Please add datasets.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        cmd = sys.argv[1].lower()

        if cmd == '--list':
            list_kaggle_datasets()
        elif cmd == '--check':
            quick_check()
        elif cmd == '--verify':
            verify_datasets()
        elif cmd == '--help':
            print("Kaggle Dataset Organizer Commands:")
            print("  python kaggle_organize_datasets.py              # Run main workflow")
            print("  python kaggle_organize_datasets.py --list       # List datasets")
            print("  python kaggle_organize_datasets.py --check      # Quick check")
            print("  python kaggle_organize_datasets.py --verify     # Verify copied datasets")
    else:
        main()
import sys
import re
from pathlib import Path
from inc.settings import settings

def delete_frame_files(dataset_root, frame_start, frame_end=None):
    # Resolve the dataset root path
    dataset_root = Path(dataset_root).resolve()
    if not dataset_root.is_dir():
        print(f"Error: Dataset root '{dataset_root}' does not exist or is not a directory.")
        return

    # Determine the range of frames to delete
    frame_end = frame_end or frame_start
    frames_to_delete = range(frame_start, frame_end + 1)

    # Compile the regex for extracting frame numbers
    frame_regex = re.compile(settings['frame_number_regex'])

    # Define subdirectories based on settings
    subdirs = {
        "images": settings['images_dir'],
        "labels": settings['labels_dir'],
        "tracks": settings['tracks_dir'],
        "orientations": settings['orientations_dir'],
    }

    # Iterate over subdirectories and delete matching files
    for subdir_name, subdir_path in subdirs.items():
        print(f"Processing subdirectory: {subdir_name}")
        subdir = dataset_root / subdir_path
        if not subdir.is_dir():
            print(f"Warning: Subdirectory '{subdir}' does not exist. Skipping.")
            continue

        for file_path in subdir.iterdir():
            match = frame_regex.search(file_path.name)
            if match:
                frame_number = int(match.group(1))
                if frame_number in frames_to_delete:
                    file_path.unlink()
                    print(f"Deleted: {file_path}")

    print("Frame deletion completed.")

if __name__ == "__main__":
    # Parse command-line arguments
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python remove_frames_from_dataset.py <dataset_root> <frame_start> [frame_end]")
        sys.exit(1)

    dataset_root = sys.argv[1]
    frame_start = int(sys.argv[2])
    frame_end = int(sys.argv[3]) if len(sys.argv) == 4 else None

    delete_frame_files(dataset_root, frame_start, frame_end)
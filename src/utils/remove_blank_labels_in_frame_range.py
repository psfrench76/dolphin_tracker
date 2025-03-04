"""
This script removes empty label and track files in a specified dataset folder within a given frame range (inclusive).
This is useful when managing background examples. It is best to remove blank labels in ranges where dolphins are
common, in order to avoid penalizing detection of true positives when the ground truth isn't labeled.

Usage: remove_blank_labels_in_frame_range.py <dataset_root_dir> <start_frame> <end_frame>
"""

import argparse
import re
from pathlib import Path
from inc.settings import settings


# Remove empty label and track files in dataset_root_dir within the frame range start_frame to end_frame (inclusive).
def _remove_blank_labels(dataset_root_dir, start_frame, end_frame):
    dataset_root_path = Path(dataset_root_dir)
    if dataset_root_path.name in [settings['images_dir'], settings['tracks_dir'], settings['labels_dir']]:
        raise ValueError("Destination directory should be the dataset root, not images, labels, or tracks directory.")

    labels_dir_path = dataset_root_path / settings['labels_dir']
    tracks_dir_path = dataset_root_path / settings['tracks_dir']

    removed_labels_count = 0
    removed_tracks_count = 0
    non_blank_labels_count = 0
    non_blank_tracks_count = 0

    # Define the pattern to extract frame number
    frame_id_pattern = re.compile(r'_(\d+)\.txt')

    # Iterate over files in the label folder
    for file_path in labels_dir_path.iterdir():
        if file_path.suffix == '.txt':
            match = frame_id_pattern.search(file_path.name)
            if match:
                frame_number = int(match.group(1))
                if start_frame <= frame_number <= end_frame:
                    if file_path.stat().st_size == 0:
                        file_path.unlink()
                        print(f"Removed: {file_path}")
                        removed_labels_count += 1
                    else:
                        non_blank_labels_count += 1

    # Iterate over files in the tracks folder
    for file_path in tracks_dir_path.iterdir():
        if file_path.suffix == '.txt':
            match = frame_id_pattern.search(file_path.name)
            if match:
                frame_number = int(match.group(1))
                if start_frame <= frame_number <= end_frame:
                    if file_path.stat().st_size == 0:
                        file_path.unlink()
                        print(f"Removed: {file_path}")
                        removed_tracks_count += 1
                    else:
                        non_blank_tracks_count += 1

    print(f"Total label files removed: {removed_labels_count}")
    print(f"Total track files removed: {removed_tracks_count}")
    print(f"Non-blank label files kept: {non_blank_labels_count}")
    print(f"Non-blank track files kept: {non_blank_tracks_count}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Remove empty label and track files in a specified dataset folder within a given frame range ("
                    "inclusive).")
    parser.add_argument('dataset_root_dir', type=Path, help='Path to the label folder')
    parser.add_argument('start_frame', type=int, help='Start frame number (inclusive)')
    parser.add_argument('end_frame', type=int, help='End frame number (inclusive)')
    args = parser.parse_args()

    _remove_blank_labels(args.dataset_root_dir, args.start_frame, args.end_frame)

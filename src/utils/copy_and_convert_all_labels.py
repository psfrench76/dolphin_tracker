"""
This script converts all label files present in the source directory itself (not any subdirectories)
and saves them to the destination directory. It uses the convert_and_save_label function from data_conversion.py.
Blank labels are no longer tracked.

Usage: copy_and_convert_all_labels.py <source_dir> <dest_dir> [--oriented_bbox]
"""

import argparse
from pathlib import Path
from inc.data_conversion import convert_and_save_label, print_run_stats
from inc.settings import settings

def copy_and_convert_all_labels(source_dir, dest_dir, oriented_bbox=False):
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    run_stats = {}

    # Sanity check for destination directory
    if dest_path.name in [settings['images_dir'], settings['tracks_dir'], settings['labels_dir']]:
        raise ValueError("Destination directory should be the dataset root, not images, labels, or tracks directory.")

    converted_count = 0

    for json_file in source_path.glob('*.json'):
        frame_stats = convert_and_save_label(json_file, dest_path, oriented_bbox)
        converted_count += 1
        for key, value in frame_stats.items():
            if key not in run_stats:
                run_stats[key] = {json_file.stem: value}
            else:
                run_stats[key][json_file.stem] = value

    print_run_stats(run_stats)
    print(f"Total frames converted: {converted_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert all labels from source directory to dataset root directory.")
    parser.add_argument('source_dir', type=str, help='Path to the source directory')
    parser.add_argument('dest_dir', type=str, help='Path to the dataset root directory')
    parser.add_argument('--oriented_bbox', action='store_true', help='Convert to oriented bounding box format')
    args = parser.parse_args()

    copy_and_convert_all_labels(args.source_dir, args.dest_dir, args.oriented_bbox)
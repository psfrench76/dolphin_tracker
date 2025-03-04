"""
This script converts label files which are present in both the source and destination directories.
It indexes the full contents of the source directory and subdirectories at the start for efficiency.
It reports any files which are present in the destination but not in the source, and notes whether they are empty labels (good) or populated (bad).

Usage: reconvert_labels_from_json.py <source_dir> <dest_dir> [--oriented_bbox]
"""

import argparse
from pathlib import Path
from inc.data_conversion import convert_and_save_label, print_run_stats, create_background_tracks_file
from inc.settings import settings

def index_files(directory):
    """
    Index all files in the directory and its subdirectories.
    """
    file_index = {}
    for file in Path(directory).rglob('*.json'):
        if file.is_file():
            file_index[file.stem] = file
    return file_index

def reconvert_labels_from_json(source_dir, dest_dir, oriented_bbox=False):
    source_path = Path(source_dir)
    dest_path = Path(dest_dir)
    run_stats = {}

    # Sanity check for destination directory
    if dest_path.name in [settings['images_dir'], settings['tracks_dir'], settings['labels_dir']]:
        raise ValueError("Destination directory should be the dataset root, not images, labels, or tracks directory.")

    source_files = index_files(source_path)
    labels_path = dest_path / settings['labels_dir']

    converted_count = 0
    background_labels_count = 0
    populated_missing_files = []

    for text_file in labels_path.glob('*.txt'):
        if text_file.stem in source_files:
            frame_stats = convert_and_save_label(source_files[text_file.stem], dest_path, oriented_bbox)
            converted_count += 1
            for key, value in frame_stats.items():
                if key not in run_stats:
                    run_stats[key] = {text_file.stem: value}
                else:
                    run_stats[key][text_file.stem] = value
        else:
            if text_file.stat().st_size == 0:
                background_labels_count += 1
                create_background_tracks_file(text_file)
            else:
                populated_missing_files.append(text_file.name)

    print_run_stats(run_stats)

    print(f"Total frames converted: {converted_count}")
    print(f"Total background label files without jsons: {background_labels_count}")
    if populated_missing_files:
        print("Labels present in destination dataset but not in source directory:")
        for file in populated_missing_files:
            print(file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconvert labels from source to dataset root directory.")
    parser.add_argument('source_dir', type=str, help='Path to the source directory')
    parser.add_argument('dest_dir', type=str, help='Path to the dataset root directory')
    parser.add_argument('--oriented_bbox', action='store_true', help='Convert to oriented bounding box format')
    args = parser.parse_args()

    reconvert_labels_from_json(args.source_dir, args.dest_dir, args.oriented_bbox)
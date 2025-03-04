"""
This script converts label files which are present in both the source and destination directories.
It indexes the full contents of the source directory and subdirectories at the start for efficiency.
It reports any files which are present in the destination but not in the source, and notes whether they are empty
labels (good) or populated (bad).

Usage: reconvert_labels_from_json.py <json_source_dir> <dataset_root_dir> [--oriented_bbox]
"""

import argparse
from pathlib import Path
from inc.data_conversion import convert_and_save_label, print_run_stats, create_background_tracks_file
from inc.settings import settings


# Finds all json files in the source directory and its subdirectories, and indexes them by filename so that they can
# be quickly accessed by the label conversion function.
def _index_files(directory):
    """
    Index all files in the directory and its subdirectories.
    """
    file_index = {}
    for file in Path(directory).rglob('*.json'):
        if file.is_file():
            file_index[file.stem] = file
    return file_index


# For every label file which is present in dataset_root_dir / settings['labels_dir'], it checks if the corresponding
# json file is present in json index, and reconverts from json to yolo format. Useful when the conversion protocol
# has changed, or a non-oriented dataset needs to be converted to oriented, etc.
def _reconvert_labels_from_json(json_source_dir, dataset_root_dir, oriented_bbox=False):
    json_source_path = Path(json_source_dir)
    dataset_root_path = Path(dataset_root_dir)
    run_stats = {}

    # Sanity check for destination directory
    if dataset_root_path.name in [settings['images_dir'], settings['tracks_dir'], settings['labels_dir']]:
        raise ValueError("Destination directory should be the dataset root, not images, labels, or tracks directory.")

    json_paths_index = _index_files(json_source_path)
    labels_dir_path = dataset_root_path / settings['labels_dir']

    converted_count = 0
    background_labels_count = 0
    populated_missing_files = []

    for label_file in labels_dir_path.glob('*.txt'):
        if label_file.stem in json_paths_index:
            frame_stats = convert_and_save_label(json_paths_index[label_file.stem], dataset_root_path, oriented_bbox)
            converted_count += 1
            for key, value in frame_stats.items():
                if key not in run_stats:
                    run_stats[key] = {label_file.stem: value}
                else:
                    run_stats[key][label_file.stem] = value
        else:
            if label_file.stat().st_size == 0:
                background_labels_count += 1
                create_background_tracks_file(label_file)  # Create a background tracks file for empty labels
            else:
                populated_missing_files.append(label_file.name)

    print_run_stats(run_stats)

    print(f"Total frames converted: {converted_count}")
    print(f"Total background label files without jsons: {background_labels_count}")
    if populated_missing_files:
        print("WARNING: Labels present in destination dataset but not in source directory:")
        for file in populated_missing_files:
            print(file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reconvert labels from source to dataset root directory.")
    parser.add_argument('json_source_dir', type=str, help='Path to the source directory')
    parser.add_argument('dataset_root_dir', type=str, help='Path to the dataset root directory')
    parser.add_argument('--oriented_bbox', action='store_true', help='Convert to oriented bounding box format')
    args = parser.parse_args()

    _reconvert_labels_from_json(args.json_source_dir, args.dataset_root_dir, args.oriented_bbox)

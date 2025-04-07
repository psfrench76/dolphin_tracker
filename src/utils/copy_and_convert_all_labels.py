"""
This script converts all label files present in the source directory itself (not any subdirectories)
and saves them to the destination directory. It uses the convert_and_save_label function from data_conversion.py.

Usage: copy_and_convert_all_labels.py <json_source_dir> <dataset_root_dr> [--oriented_bbox]
"""

import argparse
from pathlib import Path
from inc.data_conversion import convert_and_save_label, print_run_stats
from inc.settings import settings


# Copies and converts all labels present in the source directory (a directory containing json files) to the
# destination dataset, in a labels subdirectory, in yolo format.
def copy_and_convert_all_labels(json_source_dir, dataset_root_dr, oriented_bbox=False, xy_orientations=False):
    json_source_path = Path(json_source_dir)
    dataset_root_path = Path(dataset_root_dr)
    run_stats = {}

    # Sanity check for destination directory
    if dataset_root_path.name in [settings['images_dir'], settings['tracks_dir'], settings['labels_dir']]:
        raise ValueError("Destination directory should be the dataset root, not images, labels, or tracks directory.")

    converted_count = 0

    for json_file in json_source_path.glob('*.json'):
        frame_stats = convert_and_save_label(json_file, dataset_root_path, oriented_bbox, xy_orientations)
        converted_count += 1
        for key, value in frame_stats.items():
            if key == 'unrecognized_shape_labels':
                if key not in run_stats:
                    run_stats[key] = value
                else:
                    for label, count in value.items():
                        if label not in run_stats[key]:
                            run_stats[key][label] = count
                        else:
                            run_stats[key][label] += count
            else:
                if key not in run_stats:
                    run_stats[key] = {json_file.stem: value}
                else:
                    run_stats[key][json_file.stem] = value

    print_run_stats(run_stats)
    print(f"Total frames converted: {converted_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert all labels from source directory to dataset root directory.")
    parser.add_argument('json_source_dir', type=str, help='Path to the source directory')
    parser.add_argument('dataset_root_dr', type=str, help='Path to the dataset root directory')
    parser.add_argument('--oriented_bbox', action='store_true', help='Convert to oriented bounding box format')
    parser.add_argument('--xy_orientations', action='store_true', help='Convert to xy orientations format')
    args = parser.parse_args()

    copy_and_convert_all_labels(args.json_source_dir, args.dataset_root_dr, args.oriented_bbox, args.xy_orientations)

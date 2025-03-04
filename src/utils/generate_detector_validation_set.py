"""
This script generates a detector validation set from a given test set directory.
It takes an input test set directory and an output valid set directory, ensuring that both directory names are in settings['dataset_split_dirs'].
The script performs the following steps:
1. Validates the input and output directory names against settings['dataset_split_dirs'].
2. Creates the output directories if they do not exist.
3. Copies image and label files from the input directory to the output directory, skipping any image files that do not have corresponding labels.
4. Tracks and reports the number of files of each type copied.

Usage: generate_detector_validation_set.py <input_dir> <output_dir>
"""

import argparse
from pathlib import Path
import shutil
from settings import settings

def validate_directory_name(directory, valid_names):
    if not any(directory.name == name for name in valid_names):
        raise ValueError(f"The directory name must be one of {valid_names}")

def generate_detector_validation_set(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    validate_directory_name(input_path, settings['dataset_split_dirs'])
    validate_directory_name(output_path, settings['dataset_split_dirs'])

    images_input_path = input_path / settings['images_dir']
    labels_input_path = input_path / settings['labels_dir']
    images_output_path = output_path / settings['images_dir']
    labels_output_path = output_path / settings['labels_dir']

    images_output_path.mkdir(parents=True, exist_ok=True)
    labels_output_path.mkdir(parents=True, exist_ok=True)

    copied_images_count = 0
    copied_labels_count = 0
    skipped_images_count = 0

    for image_file in images_input_path.glob('*.jpg'):
        label_file = labels_input_path / (image_file.stem + '.txt')
        if label_file.exists():
            shutil.copy(image_file, images_output_path / image_file.name)
            shutil.copy(label_file, labels_output_path / label_file.name)
            copied_images_count += 1
            copied_labels_count += 1
        else:
            skipped_images_count += 1

    print(f"Total images copied: {copied_images_count}")
    print(f"Total labels copied: {copied_labels_count}")
    print(f"Total images skipped: {skipped_images_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate detector validation set from test set.")
    parser.add_argument('input_dir', type=str, help='Path to the input test set directory')
    parser.add_argument('output_dir', type=str, help='Path to the output valid set directory')
    args = parser.parse_args()

    generate_detector_validation_set(args.input_dir, args.output_dir)
"""
This script generates training and validation sets from a complete source based on a (presumably) hand-built test set.

It generates a validation set by copying only valid image/label pairs from the test set (ignoring images without
label files).

It generates a training set by copying all image/label pairs from the source (and all subdirectories) EXCEPT those
images which are present in the test set. It only includes a specified fraction of negative examples.

Usage: generate_training_and_validation_sets.py <complete_source_dir_path> <dataset_root_path>
[-n <negative_example_fraction>]

Args:
complete_source_dir_path (Path): Path to the complete source directory. This directory should contain images and
labels for all examples, preprocessed and converted into yolo format. It should also be cleaned, having bad
filenames, duplicate labels, etc. removed. See scripts copy_and_convert_all_labels.py and reconvert_labels_from_json.py
for details on this process.

dataset_root_path (Path): Path to the dataset root directory we are creating. This should already have a test set (
folder settings['dataset_test_split']) with images and labels.

negative_example_fraction (float): Proportion of negative examples to use when generating training set. This is to
cut down on training set size, since negative examples are very common in the dataset and including all of them
doesn't have a lot of value.
"""

import argparse
import shutil
import random
from pathlib import Path
from inc.settings import settings
import os


# Generate training and validation sets from COMPLETE_SOURCE_DIR_PATH and DATASET_ROOT_PATH.
def _generate_train_and_valid_sets(complete_source_dir_path, dataset_root_path, negative_example_fraction):
    if not (0 <= negative_example_fraction <= 1):
        raise ValueError("Negative example fraction must be between 0 and 1")

    if dataset_root_path.name in [settings['images_dir'], settings['tracks_dir'], settings['labels_dir']]:
        raise ValueError("Dataset directory should be the dataset root, not images, labels, or tracks directory.")

    if dataset_root_path.name in settings['dataset_split_dirs']:
        raise ValueError("Dataset directory should be the dataset root, not a split directory.")

    # Build all the paths, make directories as needed
    train_dir_path = dataset_root_path / settings['dataset_train_split']
    valid_dir_path = dataset_root_path / settings['dataset_valid_split']
    test_dir_path = dataset_root_path / settings['dataset_test_split']

    if train_dir_path.is_dir():
        raise ValueError(
            "Training directory already exists. Please remove it before running this script, because it will "
            "overwrite things chaotically.")

    if valid_dir_path.is_dir():
        raise ValueError(
            "Validation directory already exists. Please remove it before running this script, because it will "
            "overwrite things chaotically.")

    if not test_dir_path.is_dir():
        raise ValueError(f"Test directory does not exist. Please create the {settings['dataset_split']} folder and "
                         f"populate {settings['images_dir']} and {settings['labels_dir']} directories before running "
                         f"this script.")

    train_images_dir_path = train_dir_path / settings['images_dir']
    train_labels_dir_path = train_dir_path / settings['labels_dir']
    train_orientations_dir_path = train_dir_path / settings['orientations_dir']
    train_tracks_dir_path = train_dir_path / settings['tracks_dir']

    valid_images_dir_path = valid_dir_path / settings['images_dir']
    valid_labels_dir_path = valid_dir_path / settings['labels_dir']
    valid_orientations_dir_path = valid_dir_path / settings['orientations_dir']
    valid_tracks_dir_path = valid_dir_path / settings['tracks_dir']

    test_images_dir_path = test_dir_path / settings['images_dir']
    test_labels_dir_path = test_dir_path / settings['labels_dir']
    test_orientations_dir_path = test_dir_path / settings['orientations_dir']
    test_tracks_dir_path = test_dir_path / settings['tracks_dir']

    train_images_dir_path.mkdir(parents=True, exist_ok=True)
    train_labels_dir_path.mkdir(parents=True, exist_ok=True)
    train_tracks_dir_path.mkdir(parents=True, exist_ok=True)

    valid_images_dir_path.mkdir(parents=True, exist_ok=True)
    valid_labels_dir_path.mkdir(parents=True, exist_ok=True)
    valid_tracks_dir_path.mkdir(parents=True, exist_ok=True)

    if test_orientations_dir_path.exists():
        train_orientations_dir_path.mkdir(parents=True, exist_ok=True)
        valid_orientations_dir_path.mkdir(parents=True, exist_ok=True)

    test_images = set(test_images_dir_path.glob('*.jpg'))
    # Find all images in the source directory, including those behind symlinks (getting around limitation in pathlib)
    source_images = [Path(root) / file for root, _, files in
                     os.walk(complete_source_dir_path, followlinks=True) for file in files if
                     file.endswith('.jpg')]

    train_set = []
    train_set_negative = []
    valid_set = []
    images_without_labels_count = 0

    # Traverse complete_source_dir_path and find images with corresponding labels
    for image_file in source_images:
        if image_file.name not in [img.name for img in test_images]:
            label_file = image_file.parent.parent / settings['labels_dir'] / image_file.with_suffix('.txt').name
            if label_file.exists():
                orientation_file = image_file.parent.parent / settings['orientations_dir'] / image_file.with_suffix('.txt').name
                if not orientation_file.exists():
                    orientation_file = None

                track_file = image_file.parent.parent / settings['tracks_dir'] / image_file.with_suffix('.txt').name
                if not track_file.exists():
                    track_file = None

                if label_file.stat().st_size == 0:
                    train_set_negative.append((image_file, label_file, track_file, orientation_file))
                else:
                    train_set.append((image_file, label_file, track_file, orientation_file))


    negative_count = len(train_set_negative)
    # Handle negative examples based on negative_example_fraction
    if negative_example_fraction < 1:
        random.shuffle(train_set_negative)
        negative_count = int(len(train_set_negative) * negative_example_fraction)
        train_set.extend(train_set_negative[:negative_count])

    # Build validation set, a copy of test set but only examples with labels
    for image_file in test_images:
        label_file = test_labels_dir_path / (image_file.stem + '.txt')
        if label_file.exists():
            orientation_file = test_orientations_dir_path / (image_file.stem + '.txt')
            if not orientation_file.exists():
                orientation_file = None

            track_file = test_tracks_dir_path / (image_file.stem + '.txt')
            if not track_file.exists():
                track_file = None

            valid_set.append((image_file, label_file, track_file, orientation_file))
        else:
            images_without_labels_count += 1


    # Copy files to train and valid folders
    for image_path, label_path, track_path, orientation_path in train_set:
        shutil.copy(image_path, train_images_dir_path / image_path.name)
        shutil.copy(label_path, train_labels_dir_path / label_path.name)
        if track_path:
            shutil.copy(track_path, train_tracks_dir_path / track_path.name)
        if orientation_path:
            shutil.copy(orientation_path, train_orientations_dir_path / orientation_path.name)

    for image_path, label_path, track_path, orientation_path in valid_set:
        shutil.copy(image_path, valid_images_dir_path / image_path.name)
        shutil.copy(label_path, valid_labels_dir_path / label_path.name)
        if track_path:
            shutil.copy(track_path, valid_tracks_dir_path / track_path.name)
        if orientation_path:
            shutil.copy(orientation_path, valid_orientations_dir_path / orientation_path.name)


    total_dataset_size = len(train_set) + len(test_images)
    print(f"Total images in source directory: {len(source_images)}")
    print(f"Total images in test set: {len(test_images)}\n")

    print(f"Images and labels copied from test to validation set: {len(valid_set)}")
    print(f"Tracks copied from test to validation set: {len([t for _, _, t, _ in valid_set if t])}")
    print(f"Orientations copied from test to validation set: {len([o for _, _, _, o in valid_set if o])}")
    print(f"Images in test set without labels, not copied to validation set: {images_without_labels_count}\n")

    print(f"Total images and labels copied to train set: {len(train_set)}")
    print(f"Total tracks copied to train set: {len([t for _, _, t, _ in train_set if t])}")
    print(f"Total orientations copied to train set: {len([o for _, _, _, o in train_set if o])}")
    print(f"Positive images copied to train set: {len(train_set) - negative_count}")
    print(f"Negative images copied to train set: {negative_count} out of {len(train_set_negative)} available\n")

    print(f"This results in a train/test split of {len(train_set) / total_dataset_size:.2%}/"
          f"{len(test_images) / total_dataset_size:.2%}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Generate train and valid sets from input folders based on given proportions.")
    parser.add_argument('complete_source_dir_path', type=Path, help='Path to the complete source directory')
    parser.add_argument('dataset_root_path', type=Path, help='Path to the dataset root directory we are creating')
    parser.add_argument('--negative_example_fraction', '-n', default=1.0, type=float,
                        help='Proportion of negative examples to use when generating training set')
    args = parser.parse_args()

    _generate_train_and_valid_sets(args.complete_source_dir_path, args.dataset_root_path,
                                   args.negative_example_fraction)

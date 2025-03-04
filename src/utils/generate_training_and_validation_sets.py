import argparse
import shutil
import random
from pathlib import Path
from inc.settings import settings

def generate_train_and_valid_sets(dataset_root_path, complete_source_dir_path, negative_example_fraction):
    if not (0 <= negative_example_fraction <= 1):
        raise ValueError("Negative example fraction must be between 0 and 1")

    if dataset_root_path.name in [settings['images_dir'], settings['tracks_dir'], settings['labels_dir']]:
        raise ValueError("Dataset directory should be the dataset root, not images, labels, or tracks directory.")

    if dataset_root_path.name in settings['dataset_split_dirs']:
        raise ValueError("Dataset directory should be the dataset root, not a split directory.")

    train_dir_path = dataset_root_path / settings['dataset_train_split']
    valid_dir_path = dataset_root_path / settings['dataset_valid_split']
    test_dir_path = dataset_root_path / settings['dataset_test_split']

    train_images_dir_path = train_dir_path / settings['images_dir']
    train_labels_dir_path = train_dir_path / settings['labels_dir']

    valid_images_dir_path = valid_dir_path / settings['images_dir']
    valid_labels_dir_path = valid_dir_path / settings['labels_dir']

    test_images_dir_path = test_dir_path / settings['images_dir']
    test_labels_dir_path = test_dir_path / settings['labels_dir']

    train_images_dir_path.mkdir(parents=True, exist_ok=True)
    train_labels_dir_path.mkdir(parents=True, exist_ok=True)

    valid_images_dir_path.mkdir(parents=True, exist_ok=True)
    valid_labels_dir_path.mkdir(parents=True, exist_ok=True)

    # Get list of test images
    test_images = set(test_images_dir_path.glob('*.jpg'))

    train_set = []
    train_set_negative = []
    valid_set = []
    images_without_labels_count = 0

    # Traverse complete_source_dir_path and find images with corresponding labels
    for image_file in complete_source_dir_path.rglob('*.jpg'):
        if image_file not in test_images:
            label_file = image_file.parent.parent / settings['labels_dir'] / image_file.with_suffix('.txt').name
            if label_file.exists():
                if label_file.stat().st_size == 0:
                    train_set_negative.append((image_file, label_file))
                else:
                    train_set.append((image_file, label_file))

    # Handle negative examples based on negative_prop
    if negative_example_fraction < 1:
        random.shuffle(train_set_negative)
        negative_count = int(len(train_set_negative) * negative_example_fraction)
        train_set_negative = train_set_negative[:negative_count]

    train_set.extend(train_set_negative)

    for image_file in test_images:
        label_file = test_labels_dir_path / (image_file.stem + '.txt')
        if label_file.exists():
            valid_set.append((image_file, label_file))
        else:
            images_without_labels_count += 1

    # Copy files to train and valid folders
    for image_path, label_path in train_set:
        shutil.copy(image_path, train_images_dir_path / image_path.name)
        shutil.copy(label_path, train_labels_dir_path / label_path.name)

    for image_path, label_path in valid_set:
        shutil.copy(image_path, valid_images_dir_path / image_path.name)
        shutil.copy(label_path, valid_labels_dir_path / label_path.name)

    print(f"Images copied to train set: {len(train_set)}")
    print(f"Negative images copied to train set: {len(train_set_negative)}")
    print(f"Images and labels copied from test to validation set: {len(valid_set)}")
    print(f"Images in test set without labels, not copied to validation set: {images_without_labels_count}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate train and valid sets from input folders based on given proportions.")
    parser.add_argument('dataset_root_path', type=Path, help='Path to the dataset root directory we are creating')
    parser.add_argument('complete_source_dir_path', type=Path, help='Path to the complete source directory')
    parser.add_argument('negative_example_fraction', type=float, help='Proportion of negative examples to use when generating training set')
    args = parser.parse_args()

    generate_train_and_valid_sets(args.dataset_root_path, args.complete_source_dir_path, args.negative_example_fraction)
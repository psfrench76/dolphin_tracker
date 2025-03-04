import argparse
import shutil
import random
from pathlib import Path
from inc.settings import settings

def generate_train_and_valid_sets(test_folder, train_folder, valid_folder, input_folder, negative_prop):
    if not (0 <= negative_prop <= 1):
        raise ValueError("Negative example fraction must be between 0 and 1")

    (train_folder / 'images').mkdir(parents=True, exist_ok=True)
    (train_folder / 'labels').mkdir(parents=True, exist_ok=True)

    # Get list of test images
    test_images = set((test_folder / 'images').iterdir())

    # Traverse input folder and find images with corresponding labels
    train_set = []
    negative_images = []
    for image_file in input_folder.rglob('*.jpg'):
        if image_file not in test_images:
            label_file = image_file.with_suffix('.txt').parent.parent / 'labels' / image_file.with_suffix('.txt').name
            if label_file.exists():
                if label_file.stat().st_size == 0:
                    negative_images.append((image_file, label_file))
                else:
                    train_set.append((image_file, label_file))

    # Handle negative examples based on negative_prop
    if negative_prop < 1:
        random.shuffle(negative_images)
        negative_count = int(len(negative_images) * negative_prop)
        negative_images = negative_images[:negative_count]

    train_set.extend(negative_images)

    # Copy files to train and valid folders
    for image_path, label_path in train_set:
        shutil.copy(image_path, train_folder / 'images' / image_path.name)
        shutil.copy(label_path, train_folder / 'labels' / label_path.name)

    print(f"Total images processed: {len(train_set)}")
    print(f"Images copied to train set: {len(train_set)}")
    print(f"Negative images processed: {len(negative_images)}")
    print(f"Negative images copied to train set: {len(negative_images)}")

    generate_detector_validation_set(test_folder, valid_folder)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate train and valid sets from input folders based on given proportions.")
    parser.add_argument('test_folder', type=Path, help='Path to the test folder')
    parser.add_argument('train_folder', type=Path, help='Path to the train folder')
    parser.add_argument('valid_folder', type=Path, help='Path to the valid folder')
    parser.add_argument('input_folder', type=Path, help='Path to the input folder')
    parser.add_argument('negative_prop', type=float, help='Proportion of negative examples')
    args = parser.parse_args()

    generate_train_and_valid_sets(args.test_folder, args.train_folder, args.valid_folder, args.input_folder, args.negative_prop)
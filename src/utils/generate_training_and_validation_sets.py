import argparse
import shutil
import random
from pathlib import Path

def generate_train_and_valid_sets(test_folder, train_folder, valid_folder, input_folder, train_prop, valid_prop, test_prop, negative_prop):
    """
    Generate train and valid sets from input folders based on given proportions.
    """
    # Check if proportions sum to 1
    if not (0 <= train_prop <= 1 and 0 <= valid_prop <= 1 and 0 <= test_prop <= 1 and 0 <= negative_prop <= 1):
        raise ValueError("Proportions must be between 0 and 1")
    if train_prop + valid_prop + test_prop != 1:
        raise ValueError("Proportions must sum to 1")

    # Normalize train_prop and valid_prop to sum to 1
    total_prop = train_prop + valid_prop
    train_prop /= total_prop
    valid_prop /= total_prop

    # Ensure the train and valid folders exist
    (train_folder / 'images').mkdir(parents=True, exist_ok=True)
    (train_folder / 'labels').mkdir(parents=True, exist_ok=True)
    (valid_folder / 'images').mkdir(parents=True, exist_ok=True)
    (valid_folder / 'labels').mkdir(parents=True, exist_ok=True)

    # Get list of test images
    test_images = set((test_folder / 'images').iterdir())

    # Traverse input folder and find images with corresponding labels
    images_to_process = []
    negative_images = []
    for image_file in input_folder.rglob('*.jpg'):
        if image_file not in test_images:
            label_file = image_file.with_suffix('.txt').parent.parent / 'labels' / image_file.with_suffix('.txt').name
            if label_file.exists():
                if label_file.stat().st_size == 0:
                    negative_images.append((image_file, label_file))
                else:
                    images_to_process.append((image_file, label_file))

    # Shuffle and split images into train and valid sets
    random.shuffle(images_to_process)
    train_count = int(len(images_to_process) * train_prop)
    valid_count = len(images_to_process) - train_count

    train_set = images_to_process[:train_count]
    valid_set = images_to_process[train_count:]

    # Handle negative examples based on negative_prop
    if negative_prop < 1:
        random.shuffle(negative_images)
        negative_count = int(len(negative_images) * negative_prop)
        negative_images = negative_images[:negative_count]

    # Split negative examples into train and valid sets based on proportions
    negative_train_count = int(len(negative_images) * train_prop)
    negative_valid_count = len(negative_images) - negative_train_count

    train_set.extend(negative_images[:negative_train_count])
    valid_set.extend(negative_images[negative_train_count:])

    # Copy files to train and valid folders
    for image_path, label_path in train_set:
        shutil.copy(image_path, train_folder / 'images' / image_path.name)
        shutil.copy(label_path, train_folder / 'labels' / label_path.name)

    for image_path, label_path in valid_set:
        shutil.copy(image_path, valid_folder / 'images' / image_path.name)
        shutil.copy(label_path, valid_folder / 'labels' / label_path.name)

    print(f"Total images processed: {len(images_to_process)}")
    print(f"Images copied to train set: {train_count}")
    print(f"Images copied to valid set: {valid_count}")
    print(f"Negative images processed: {len(negative_images)}")
    print(f"Negative images copied to train set: {negative_train_count}")
    print(f"Negative images copied to valid set: {negative_valid_count}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate train and valid sets from input folders based on given proportions.")
    parser.add_argument('test_folder', type=Path, help='Path to the test folder')
    parser.add_argument('train_folder', type=Path, help='Path to the train folder')
    parser.add_argument('valid_folder', type=Path, help='Path to the valid folder')
    parser.add_argument('input_folder', type=Path, help='Path to the input folder')
    parser.add_argument('train_prop', type=float, help='Proportion of training data')
    parser.add_argument('valid_prop', type=float, help='Proportion of validation data')
    parser.add_argument('test_prop', type=float, help='Proportion of test data')
    parser.add_argument('negative_prop', type=float, help='Proportion of negative examples')
    args = parser.parse_args()

    generate_train_and_valid_sets(args.test_folder, args.train_folder, args.valid_folder, args.input_folder, args.train_prop, args.valid_prop, args.test_prop, args.negative_prop)
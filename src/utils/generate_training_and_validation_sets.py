import os
import shutil
import random
import click


@click.command()
@click.argument('test_folder', type=click.Path(exists=True))
@click.argument('train_folder', type=click.Path())
@click.argument('valid_folder', type=click.Path())
@click.argument('input_folder', type=click.Path(exists=True))
@click.argument('train_prop', type=float)
@click.argument('valid_prop', type=float)
@click.argument('test_prop', type=float)
@click.argument('negative_prop', type=float)
def generate_train_and_valid_sets(test_folder, train_folder, valid_folder, input_folder, train_prop, valid_prop,
                                  test_prop, negative_prop):
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

    # Convert paths to strings
    test_folder = str(test_folder)
    train_folder = str(train_folder)
    valid_folder = str(valid_folder)
    input_folder = str(input_folder)

    # Ensure the train and valid folders exist
    os.makedirs(os.path.join(train_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_folder, 'labels'), exist_ok=True)
    os.makedirs(os.path.join(valid_folder, 'images'), exist_ok=True)
    os.makedirs(os.path.join(valid_folder, 'labels'), exist_ok=True)

    # Get list of test images
    test_images = set(os.listdir(os.path.join(test_folder, 'images')))

    # Traverse input folder and find images with corresponding labels
    images_to_process = []
    negative_images = []
    for root, _, files in os.walk(input_folder, followlinks=True):
        for file in files:
            if file.endswith('.jpg') and file not in test_images:
                label_path = root.replace('images', 'labels')
                label_file = os.path.join(label_path, os.path.splitext(file)[0] + '.txt')
                if os.path.exists(label_file):
                    if os.path.getsize(label_file) == 0:
                        negative_images.append((os.path.join(root, file), label_file))
                    else:
                        images_to_process.append((os.path.join(root, file), label_file))

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
        image_filename = os.path.basename(image_path)
        label_filename = os.path.basename(label_path)
        shutil.copy(image_path, os.path.join(train_folder, 'images', image_filename))
        shutil.copy(label_path, os.path.join(train_folder, 'labels', label_filename))

    for image_path, label_path in valid_set:
        image_filename = os.path.basename(image_path)
        label_filename = os.path.basename(label_path)
        shutil.copy(image_path, os.path.join(valid_folder, 'images', image_filename))
        shutil.copy(label_path, os.path.join(valid_folder, 'labels', label_filename))

    click.echo(f"Total images processed: {len(images_to_process)}")
    click.echo(f"Images copied to train set: {train_count}")
    click.echo(f"Images copied to valid set: {valid_count}")
    click.echo(f"Negative images processed: {len(negative_images)}")
    click.echo(f"Negative images copied to train set: {negative_train_count}")
    click.echo(f"Negative images copied to valid set: {negative_valid_count}")


if __name__ == '__main__':
    generate_train_and_valid_sets()

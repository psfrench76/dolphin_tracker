import shutil
from pathlib import Path
import argparse

def index_images(source_images_path):
    """
    Index all images in the source_images directory and its subdirectories.
    """
    image_index = {}
    for image_file in source_images_path.rglob('*.jpg'):
        image_index[image_file.stem] = image_file
    return image_index

def copy_images(input_labels, source_images):
    """
    Copy images from SOURCE_IMAGES to a sibling 'images' directory of INPUT_LABELS.
    The images must have the same filename stem as the labels, but with the .jpg extension.
    """
    input_labels_path = Path(input_labels)
    source_images_path = Path(source_images)
    images_output_path = input_labels_path.parent / 'images'
    images_output_path.mkdir(exist_ok=True)

    # Index images
    image_index = index_images(source_images_path)

    copied_files_count = 0
    missing_files_count = 0

    for label_file in input_labels_path.iterdir():
        if label_file.is_file():
            image_stem = label_file.stem
            image_file = image_index.get(image_stem)

            if image_file and image_file.exists():
                shutil.copy(image_file, images_output_path / image_file.name)
                copied_files_count += 1
            else:
                missing_files_count += 1

    print(f"Total files copied: {copied_files_count}")
    print(f"Total files not found: {missing_files_count}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Copy images based on label filenames.")
    parser.add_argument('input_labels', type=str, help='Path to the input labels directory')
    parser.add_argument('source_images', type=str, help='Path to the source images directory')
    args = parser.parse_args()

    copy_images(args.input_labels, args.source_images)
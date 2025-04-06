import argparse
from pathlib import Path
from inc.orientation_io import crop_image_with_bbox
from inc.settings import settings

def main():
    parser = argparse.ArgumentParser(description="Generate a cropped image sample based on a label index.")
    parser.add_argument('image_file', type=Path, help="Path to the image file.")
    parser.add_argument('label_index', type=int, help="Label index (0-based).")
    parser.add_argument('--output_folder', '-o', type=Path, default=Path.cwd(), help="Optional output folder. Defaults to current working directory.")
    args = parser.parse_args()

    image_file = args.image_file
    label_index = args.label_index
    output_folder = args.output_folder

    dataset_root_dir = image_file.parent.parent
    label_dir = dataset_root_dir / settings['labels_dir']
    label_file = label_dir / f"{image_file.stem}.txt"

    if not label_file.exists():
        raise FileNotFoundError(f"Label file {label_file} not found.")

    # Read the label from the label file
    with open(label_file, 'r') as f:
        lines = f.readlines()
        if label_index >= len(lines):
            raise IndexError(f"Label index {label_index} out of range for file {label_file}.")
        label_line = lines[label_index].strip().split()
        bbox = list(map(float, label_line[1:5]))

    # Generate the cropped image
    cropped_image = crop_image_with_bbox(image_file, bbox, 'yolo')

    # Create the output file path
    output_file = output_folder / f"{image_file.stem}_label_{label_index}_cropped.jpg"

    # Save the cropped image
    cropped_image.save(output_file)
    print(f"Cropped image saved to {output_file}")

if __name__ == "__main__":
    main()
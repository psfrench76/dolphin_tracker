import argparse
from pathlib import Path
import numpy as np
from PIL import Image
import subprocess

def get_image_size(image_path):
    try:
        result = subprocess.run(["file", str(image_path)], capture_output=True, text=True, check=True)
        output = result.stdout
        dimensions = next(part for part in output.split(",") if "x" in part and part.strip().replace("x", "").isdigit())
        width, height = map(int, dimensions.strip().split("x"))
        return width, height
    except Exception as e:
        raise ValueError(f"Error extracting image size for {image_path}: {e}")

def calculate_label_statistics(dataset_dir, yolo_output_file=None):
    image_dir = Path(dataset_dir) / "images"
    label_dir = Path(dataset_dir) / "labels"

    widths = []
    heights = []
    yolo_areas = []
    pixel_widths = []
    pixel_heights = []
    pixel_areas = []

    if yolo_output_file:
        with open(yolo_output_file, "r") as f:
            lines = f.readlines()
    else:
        lines = []
        for label_file in label_dir.glob("*.txt"):
            with open(label_file, "r") as f:
                for line in f:
                    lines.append(f"{label_file.stem}.jpg {line.strip()}")

    for line in lines:
        parts = line.strip().split()
        if len(parts) != 6:
            continue  # Skip malformed lines
        image_name, class_id, x, y, width, height = parts
        width, height = float(width), float(height)

        image_file = image_dir / image_name
        if not image_file.exists():
            print(f"Warning: Image file {image_file} not found for label.")
            continue

        try:
            img_width, img_height = get_image_size(image_file)
        except Exception as e:
            print(f"Error reading image size for {image_file}: {e}")
            continue

        widths.append(width)
        heights.append(height)
        yolo_areas.append(width * height)  # YOLO area
        pixel_width = width * img_width
        pixel_height = height * img_height
        pixel_widths.append(pixel_width)
        pixel_heights.append(pixel_height)
        pixel_areas.append(pixel_width * pixel_height)

    if not widths or not heights:
        print("No labels found.")
        return

    mean_width = np.mean(widths)
    std_width = np.std(widths)
    mean_height = np.mean(heights)
    std_height = np.std(heights)

    mean_yolo_area = np.mean(yolo_areas)
    std_yolo_area = np.std(yolo_areas)

    mean_pixel_width = np.mean(pixel_widths)
    std_pixel_width = np.std(pixel_widths)
    mean_pixel_height = np.mean(pixel_heights)
    std_pixel_height = np.std(pixel_heights)

    mean_pixel_area = np.mean(pixel_areas)
    std_pixel_area = np.std(pixel_areas)

    # Calculate standard deviations as percentages of the mean
    std_width_pct = (std_width / mean_width) * 100 if mean_width != 0 else 0
    std_height_pct = (std_height / mean_height) * 100 if mean_height != 0 else 0
    std_yolo_area_pct = (std_yolo_area / mean_yolo_area) * 100 if mean_yolo_area != 0 else 0
    std_pixel_width_pct = (std_pixel_width / mean_pixel_width) * 100 if mean_pixel_width != 0 else 0
    std_pixel_height_pct = (std_pixel_height / mean_pixel_height) * 100 if mean_pixel_height != 0 else 0
    std_pixel_area_pct = (std_pixel_area / mean_pixel_area) * 100 if mean_pixel_area != 0 else 0

    print(f"Mean Width (YOLO): {mean_width:.6f}, Standard Deviation: {std_width:.6f} ({std_width_pct:.2f}%)")
    print(f"Mean Height (YOLO): {mean_height:.6f}, Standard Deviation: {std_height:.6f} ({std_height_pct:.2f}%)")
    print(f"Mean Area (YOLO): {mean_yolo_area:.6f}, Standard Deviation: {std_yolo_area:.6f} ({std_yolo_area_pct:.2f}%)")
    print(f"Mean Width (Pixels): {mean_pixel_width:.2f}, Standard Deviation: {std_pixel_width:.2f} ({std_pixel_width_pct:.2f}%)")
    print(f"Mean Height (Pixels): {mean_pixel_height:.2f}, Standard Deviation: {std_pixel_height:.2f} ({std_pixel_height_pct:.2f}%)")
    print(f"Mean Area (Pixels): {mean_pixel_area:.2f}, Standard Deviation: {std_pixel_area:.2f} ({std_pixel_area_pct:.2f}%)")

def main():
    parser = argparse.ArgumentParser(description="Calculate statistics for YOLO labels.")
    parser.add_argument("dataset_dir", type=str, help="Path to the root dataset directory containing 'images' and 'labels' subdirectories.")
    parser.add_argument("--yolo_output_file", type=str, default=None, help="Path to the YOLO output file containing all labels.")
    args = parser.parse_args()

    calculate_label_statistics(args.dataset_dir, args.yolo_output_file)

if __name__ == "__main__":
    main()
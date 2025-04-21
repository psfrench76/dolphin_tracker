import argparse
import random
from pathlib import Path
from PIL import Image
import shutil
import numpy as np
from tqdm import tqdm


def process_image(image_path, zoom, centered):
    image = Image.open(image_path)
    original_width, original_height = image.size

    new_width = int(original_width * zoom)
    new_height = int(original_height * zoom)

    # Approximate average color by resizing to 1x1
    avg_color = image.resize((1, 1), Image.Resampling.BOX).getpixel((0, 0))
    new_image = Image.new("RGB", (original_width, original_height), avg_color)

    # Use a faster resizing method
    resized_image = image.resize((new_width, new_height), Image.Resampling.BOX)

    if centered:
        x_offset = (original_width - new_width) // 2
        y_offset = (original_height - new_height) // 2
    else:
        x_offset = random.randint(0, original_width - new_width)
        y_offset = random.randint(0, original_height - new_height)

    new_image.paste(resized_image, (x_offset, y_offset))

    return new_image, x_offset, y_offset, original_width, original_height


def process_label(label_path, zoom, x_offset, y_offset, original_width, original_height):
    with open(label_path, "r") as f:
        lines = f.readlines()

    new_labels = []
    for line in lines:
        parts = line.strip().split()
        class_id, x, y, w, h = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        pct_offset = (1 - zoom) / 2
        x = x * zoom + pct_offset
        y = y * zoom + pct_offset
        # Adjust w and h for zoom
        w *= zoom
        h *= zoom

        new_labels.append(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    return new_labels


def process_orientation(orientation_path, zoom):
    with open(orientation_path, "r") as f:
        lines = f.readlines()

    new_orientations = []
    for line in lines:
        parts = line.strip().split()
        label_index, x, y = parts[0], float(parts[1]), float(parts[2])

        x *= zoom
        y *= zoom

        new_orientations.append(f"{label_index} {x:.6f} {y:.6f}")

    return new_orientations


import concurrent.futures

def process_file(file_path, source_dir, dest_dir, zoom, centered):
    file_counts = {"images": 0, "labels": 0, "orientations": 0, "tracks": 0}
    ignored_files = 0

    if file_path.suffix == ".cache":
        ignored_files += 1
        return file_counts, ignored_files

    relative_path = file_path.relative_to(source_dir)
    dest_path = dest_dir / relative_path

    if file_path.is_dir():
        dest_path.mkdir(parents=True, exist_ok=True)
    elif "images" in str(file_path.parent):
        new_image, x_offset, y_offset, original_width, original_height = process_image(file_path, zoom, centered)
        new_image.save(dest_path)
        file_counts["images"] += 1

        label_path = file_path.parent.parent / "labels" / file_path.with_suffix(".txt").name
        if label_path.exists():
            new_labels = process_label(label_path, zoom, x_offset, y_offset, original_width, original_height)
            dest_label_path = dest_path.parent.parent / "labels" / file_path.with_suffix(".txt").name
            dest_label_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_label_path, "w") as f:
                f.write("\n".join(new_labels))
            file_counts["labels"] += 1

        orientation_path = file_path.parent.parent / "orientations" / file_path.with_suffix(".txt").name
        if orientation_path.exists():
            new_orientations = process_orientation(orientation_path, zoom)
            dest_orientation_path = dest_path.parent.parent / "orientations" / file_path.with_suffix(".txt").name
            dest_orientation_path.parent.mkdir(parents=True, exist_ok=True)
            with open(dest_orientation_path, "w") as f:
                f.write("\n".join(new_orientations))
            file_counts["orientations"] += 1
    elif "tracks" in str(file_path.parent):
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(file_path, dest_path)
        file_counts["tracks"] += 1

    return file_counts, ignored_files


def process_dataset(source_dir, dest_dir, zoom, centered):
    file_counts = {"images": 0, "labels": 0, "orientations": 0, "tracks": 0}
    ignored_files = 0

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_file, file_path, source_dir, dest_dir, zoom, centered)
            for file_path in source_dir.rglob("*")
        ]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing dataset"):
            result_counts, result_ignored = future.result()
            for key in file_counts:
                file_counts[key] += result_counts[key]
            ignored_files += result_ignored

    print("\nFile counts:")
    for file_type, count in file_counts.items():
        print(f"{file_type.capitalize()}: {count}")
    print(f"Ignored files: {ignored_files}")


def main():
    parser = argparse.ArgumentParser(description="Dataset management script with zoom functionality.")
    parser.add_argument("source_dir", type=Path, help="Path to the source dataset directory.")
    parser.add_argument("dest_dir", type=Path, help="Path to the destination dataset directory.")
    parser.add_argument("zoom", type=float, help="Zoom factor to apply to the dataset.")
    parser.add_argument("--centered", action="store_true", help="Center the shrunk image in the new image.")

    args = parser.parse_args()

    if args.zoom >= 1:
        raise ValueError("Zoom factor must be less than 1 for now.")

    process_dataset(args.source_dir, args.dest_dir, args.zoom, args.centered)


if __name__ == "__main__":
    main()
import numpy as np
from PIL import Image
from pathlib import Path
import argparse
import random
from inc.settings import settings
import time
from tqdm import tqdm

def get_average_color_sampled(image_path, sample_fraction=0.1):
    start_time = time.time()

    # Load and convert the image
    load_start = time.time()
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    load_end = time.time()

    # Calculate sample size
    total_pixels = width * height
    sample_size = int(total_pixels * sample_fraction)

    # Convert image to NumPy array
    array_start = time.time()
    image_array = np.array(image)
    array_end = time.time()

    # Generate random pixel coordinates
    coords_start = time.time()
    x_coords = np.random.randint(0, width, size=sample_size)
    y_coords = np.random.randint(0, height, size=sample_size)
    coords_end = time.time()

    # Extract sampled pixels
    sample_start = time.time()
    sampled_pixels = image_array[y_coords, x_coords]
    sample_end = time.time()

    # Compute the average color
    avg_start = time.time()
    avg_color = tuple(sampled_pixels.mean(axis=0).astype(int))
    avg_end = time.time()

    end_time = time.time()

    # Print timing information
    # print(f"Timing for {image_path.name}:")
    # print(f"  Load image: {load_end - load_start:.4f} seconds")
    # print(f"  Convert to NumPy array: {array_end - array_start:.4f} seconds")
    # print(f"  Generate random coordinates: {coords_end - coords_start:.4f} seconds")
    # print(f"  Extract sampled pixels: {sample_end - sample_start:.4f} seconds")
    # print(f"  Compute average color: {avg_end - avg_start:.4f} seconds")
    # print(f"  Total time: {end_time - start_time:.4f} seconds")

    return avg_color

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get average color of images in a dataset using sampled pixels.")
    parser.add_argument("dataset_root", type=str, help="Path to the dataset root folder.")
    parser.add_argument("--sample_fraction", type=float, default=0.01, help="Fraction of pixels to sample (0.0 to 1.0).")
    parser.add_argument("--frame_sample_fraction", type=float, default=0.01, help="Fraction of frames to sample (0.0 to 1.0).")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.is_dir():
        raise ValueError(f"Dataset root folder {dataset_root} must be a directory.")

    images_folder = dataset_root / settings['images_dir']
    if not images_folder.is_dir():
        raise ValueError(f"Images folder {images_folder} must be a directory.")

    image_paths = list(images_folder.glob("*.jpg"))
    if args.frame_sample_fraction < 1.0:
        sample_size = int(len(image_paths) * args.frame_sample_fraction)
        image_paths = random.sample(image_paths, sample_size)

    print(f"Sampling {len(image_paths)} images from {len(list(images_folder.glob('*.jpg')))} total images.")
    print(f"Sampling {args.sample_fraction * 100:.2f}% of pixels from each image.")

    average_colors = []
    for i, image_path in tqdm(enumerate(image_paths), total=len(image_paths), desc="Processing images"):
        avg_color = get_average_color_sampled(image_path, args.sample_fraction)
        average_colors.append((image_path.name, avg_color))
        # print(f"Image {i} of {len(image_paths)}: {image_path.name}: {avg_color}")

    # Average all image colors together and display the result
    total_avg_color = tuple(
        sum(color[i] for _, color in average_colors) // len(average_colors)
        for i in range(3)
    )
    hex_color = "#{:02x}{:02x}{:02x}".format(*total_avg_color)
    print(f"Average color of the dataset: {total_avg_color} (Hex: {hex_color})")
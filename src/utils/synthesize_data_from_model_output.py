"""
This script generates new frames for a synthetic dataset by copying images and generating labels, tracks, and orientations files
for a specified frame range from an input dataset.

Note: The frame range is inclusive of the first frame and EXCLUSIVE of the last frame.

Usage: create_synthetic_dataset.py <image_source_folder> <track_from_video_folder> <frame_start> <frame_end>
"""

import argparse
from pathlib import Path
import shutil
import pandas as pd
from inc.settings import settings, storage_path

def generate_dataset_frames(extracted_frames_root, model_output_folder, frame_start, frame_end):
    # Validate input paths
    extracted_frames_root = Path(extracted_frames_root)
    if not extracted_frames_root.is_dir():
        raise ValueError(f"Extracted frames root folder {extracted_frames_root} must be a directory.")

    if extracted_frames_root.name in [settings['images_dir'], settings['tracks_dir'], settings['labels_dir']]:
        raise ValueError(f"Extracted frames root folder {extracted_frames_root} should be the dataset root, not images, labels, or tracks directory.")

    model_output_folder = Path(model_output_folder)
    if not model_output_folder.is_dir():
        raise ValueError(f"Model output folder {model_output_folder} must be a directory.")

    # Check for required files in model_output_folder
    results_files = list(model_output_folder.glob(f"*{settings['results_file_suffix']}"))
    orientations_files = list(model_output_folder.glob(f"*{settings['orientations_results_suffix']}"))
    images_index_files = list(model_output_folder.glob(f"*{settings['images_index_suffix']}"))
    if not results_files or not orientations_files or not images_index_files:
        raise ValueError(f"Model output folder {model_output_folder} must contain *{settings['results_file_suffix']}, *{settings['orientations_results_suffix']}, and *{settings['images_index_suffix']} files.")

    # Load dataframes
    results_df = pd.read_csv(results_files[0], names=settings['bbox_file_columns'], header=None)
    images_index_df = pd.read_csv(images_index_files[0], names=["filename"], header=None)

    # Apply Path.stem to images index filenames
    images_index_df["filename"] = images_index_df["filename"].apply(lambda x: Path(x).stem)

    orientations_df = pd.read_csv(orientations_files[0])

    # Combine results and images index dataframes
    results_df["filename"] = images_index_df["filename"]
    combined_df = pd.merge(results_df, orientations_df, left_on=["filename", "id"], right_on=["filename", "object_id"])

    # Determine dataset name and output path
    dataset_name = extracted_frames_root.name
    synthetic_dataset_path = storage_path("data/synthetic") / dataset_name
    image_source_folder = extracted_frames_root / settings['images_dir']

    # Create subfolders for the new dataset
    dest_images_path = synthetic_dataset_path / "images"
    dest_labels_path = synthetic_dataset_path / "labels"
    dest_tracks_path = synthetic_dataset_path / "tracks"
    dest_orientations_path = synthetic_dataset_path / "orientations"

    for folder in [dest_images_path, dest_labels_path, dest_tracks_path, dest_orientations_path]:
        folder.mkdir(parents=True, exist_ok=True)

    for frame_number in range(frame_start, frame_end):
        # Copy images
        print(f"Searching {image_source_folder} for image files with frame number {frame_number}")
        image_file = list(image_source_folder.glob(f"*_{frame_number:06d}.jpg"))
        if not image_file:
            print(f"No image files found for frame {frame_number}. Skipping.")
            continue
        elif len(image_file) > 1:
            print(f"Multiple image files found for frame {frame_number}. Using the first one.")
            image_file = image_file[0]
        else:
            image_file = image_file[0]

        frame_stem = image_file.stem

        if image_file.exists():
            new_image_path = dest_images_path / image_file.name
            print(f"Copying image file {image_file} to {new_image_path}")
            shutil.copy(image_file, new_image_path)
        else:
            print(f"Image file {image_file} does not exist. Skipping.")
            continue

        # Filter dataframe for the current frame_stem
        frame_df = combined_df[combined_df["filename"] == frame_stem]

        # Generate label file
        _generate_label_file(frame_df, dest_labels_path)

        # Generate track file
        _generate_track_file(frame_df, dest_tracks_path)

        # Generate orientation file
        _generate_orientation_file(frame_df, dest_orientations_path)

    print(f"Dataset frames generated at {synthetic_dataset_path}")


def _generate_label_file(frame_df, dest_folder):
    # Generate the output file path
    output_file = dest_folder / f"{frame_df['filename'].iloc[0]}.txt"

    # Create a new dataframe with the required columns
    label_data = frame_df[['x', 'y', 'w', 'h']].copy()
    label_data.insert(0, 'class', 0)  # Insert a column of 0s as the first column

    # Save the data to the file with space-delimited format and no header
    label_data.to_csv(output_file, sep=' ', header=False, index=False)

    print(f"Label file generated at {output_file}")


def _generate_track_file(frame_df, dest_folder):
    # Generate the output file path
    output_file = dest_folder / f"{frame_df['filename'].iloc[0]}.txt"

    # Save the id column to the file without header or index
    frame_df[['id']].to_csv(output_file, header=False, index=False)

    print(f"Track file generated at {output_file}")


def _generate_orientation_file(frame_df, dest_folder):
    # Generate the output file path
    output_file = dest_folder / f"{frame_df['filename'].iloc[0]}.txt"

    # Create a new dataframe with a sequential index and the required columns
    orientation_data = frame_df[['x_val', 'y_val']].copy()
    orientation_data.insert(0, 'index', range(len(orientation_data)))

    # Save the data to the file with space-delimited format and no header
    orientation_data.to_csv(output_file, sep=' ', header=False, index=False)

    print(f"Orientation file generated at {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a synthetic dataset by copying images and generating labels, tracks, and orientations for a specified frame range."
    )
    parser.add_argument("image_source_folder", type=Path, help="Path to the image source folder")
    parser.add_argument("track_from_video_folder", type=Path, help="Path to the track from video output folder")
    parser.add_argument("frame_start", type=int, help="Start frame number (inclusive)")
    parser.add_argument("frame_end", type=int, help="End frame number (exclusive)")

    args = parser.parse_args()

    generate_dataset_frames(
        args.image_source_folder,
        args.track_from_video_folder,
        args.frame_start,
        args.frame_end
    )
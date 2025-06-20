"""
This script generates new frames for a synthetic dataset by copying images and generating labels, tracks, and orientations files
for a specified frame range from an input dataset based on model output.

Note: Orientations are generated from the moving average column of the researcher output file when available. The fallback is the raw orientations output file.

Note: The frame range is INCLUSIVE of the first frame and INCLUSIVE of the last frame.

Usage: create_synthetic_dataset.py <image_source_folder> <track_from_video_folder> <frame_start> <frame_end>
"""

import argparse
from pathlib import Path
import shutil
import pandas as pd
from inc.settings import settings, storage_path

def generate_dataset_frames(extracted_frames_root, model_output_folder, frame_start, frame_end, background=False):
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
    researcher_output_files = list(model_output_folder.glob(f"*{settings['researcher_output_suffix']}"))
    images_index_files = list(model_output_folder.glob(f"*{settings['images_index_suffix']}"))
    if not results_files or not orientations_files or not images_index_files:
        raise ValueError(f"Model output folder {model_output_folder} must contain *{settings['results_file_suffix']}, *{settings['orientations_results_suffix']}, and *{settings['images_index_suffix']} files.")

    # Load dataframes
    results_df = pd.read_csv(results_files[0], names=settings['bbox_file_columns'], header=None)
    images_index_df = pd.read_csv(images_index_files[0], names=["filename"], header=None)

    # Apply Path.stem to images index filenames
    images_index_df["filename"] = images_index_df["filename"].apply(lambda x: Path(x).stem)
    results_df["filename"] = images_index_df["filename"]

    if len(researcher_output_files) > 0:
        print(f"Using researcher output file {researcher_output_files[0]}")
        researcher_output_df = pd.read_csv(researcher_output_files[0])
        researcher_output_df = researcher_output_df.rename(columns={"MovingAvgAngleXVal": "x_val", "MovingAvgAngleYVal": "y_val"})
        results_df = pd.merge(results_df, researcher_output_df, left_on=["filename", "id"], right_on=["FrameID", "ObjectID"], how="left")
    else:
        print(f"Didn't find researcher output file. Using orientations file {orientations_files[0]}")
        orientations_df = pd.read_csv(orientations_files[0])
        results_df = pd.merge(results_df, orientations_df, left_on=["filename", "id"], right_on=["filename", "object_id"])

    # Determine dataset name and output path
    dataset_name = extracted_frames_root.name
    synthetic_dataset_path = storage_path(settings['synthetic_data_dir']) / dataset_name
    image_source_folder = extracted_frames_root / settings['images_dir']

    # Create subfolders for the new dataset
    dest_images_path = synthetic_dataset_path / settings['images_dir']
    dest_labels_path = synthetic_dataset_path / settings['labels_dir']
    dest_tracks_path = synthetic_dataset_path / settings['tracks_dir']
    dest_orientations_path = synthetic_dataset_path / settings['orientations_dir']

    for folder in [dest_images_path, dest_labels_path, dest_tracks_path, dest_orientations_path]:
        folder.mkdir(parents=True, exist_ok=True)

    frames_generated = 0

    for frame_number in range(frame_start, frame_end + 1):
        # Copy images
        # print(f"Searching {image_source_folder} for image files with frame number {frame_number}")
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
            #print(f"Copying image file {image_file} to {new_image_path}")
            shutil.copy(image_file, new_image_path)
        else:
            print(f"Image file {image_file} does not exist. Skipping.")
            continue

        # Filter dataframe for the current frame_stem
        frame_df = results_df[results_df["filename"] == frame_stem]

        if not background:
            # Generate label file
            _generate_label_file(frame_stem, frame_df, dest_labels_path)

            # Generate track file
            _generate_track_file(frame_stem, frame_df, dest_tracks_path)

            # Generate orientation file
            _generate_orientation_file(frame_stem, frame_df, dest_orientations_path)
        else:
            _generate_background_file(frame_stem, dest_labels_path)
            _generate_background_file(frame_stem, dest_tracks_path)
            _generate_background_file(frame_stem, dest_orientations_path)

        frames_generated += 1

    if background:
        print(f"{frames_generated} background frames generated at {synthetic_dataset_path}")
    else:
        print(f"{frames_generated} dataset frames generated at {synthetic_dataset_path}")


def _generate_label_file(frame_stem, frame_df, dest_folder):
    # Generate the output file path
    output_file = dest_folder / f"{frame_stem}.txt"

    # Create a new dataframe with the required columns
    label_data = frame_df[['x', 'y', 'w', 'h']].copy()
    label_data.insert(0, 'class', 0)  # Insert a column of 0s as the first column

    # Save the data to the file with space-delimited format and no header
    label_data.to_csv(output_file, sep=' ', header=False, index=False)

    #print(f"Label file generated at {output_file}")


def _generate_track_file(frame_stem, frame_df, dest_folder):
    # Generate the output file path
    output_file = dest_folder / f"{frame_stem}.txt"

    # Save the id column to the file without header or index
    frame_df[['id']].to_csv(output_file, header=False, index=False)

    #print(f"Track file generated at {output_file}")


def _generate_orientation_file(frame_stem, frame_df, dest_folder):
    # Generate the output file path
    output_file = dest_folder / f"{frame_stem}.txt"

    # Create a new dataframe with a sequential index and the required columns
    orientation_data = frame_df[['x_val', 'y_val']].copy()
    orientation_data.insert(0, 'index', range(len(orientation_data)))

    # Save the data to the file with space-delimited format and no header
    orientation_data.to_csv(output_file, sep=' ', header=False, index=False)

    #print(f"Orientation file generated at {output_file}")

def _generate_background_file(frame_stem, dest_folder):
    # Generate the output file path
    output_file = dest_folder / f"{frame_stem}.txt"

    # Create an empty file to indicate no objects
    with open(output_file, 'w') as f:
        # Just an empty file for backgrounds
        pass

    #print(f"Background file generated at {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create a synthetic dataset by copying images and generating labels, tracks, and orientations for a specified frame range."
    )
    parser.add_argument("image_source_folder", type=Path, help="Path to the image source folder")
    parser.add_argument("track_from_video_folder", type=Path, help="Path to the track from video output folder")
    parser.add_argument("frame_start", type=int, help="Start frame number (inclusive)")
    parser.add_argument("frame_end", type=int, help="End frame number (inclusive)")
    parser.add_argument("--background", "-b", action="store_true", help="Use these frames as backgrounds in the dataset")

    args = parser.parse_args()

    generate_dataset_frames(
        args.image_source_folder,
        args.track_from_video_folder,
        args.frame_start,
        args.frame_end,
        args.background
    )
"""
This module contains a function to generate videos with bounding boxes and labels from a dataset. The function
requires a dataset root directory, which should contain an images (settings['images_dir']) directory. If a bbox_file
path is provided, then it should be in MOT15 format, as an output from the tracker. If no bbox_file is provided,
then the function will look for labels and tracks in the dataset root directory. The output video will be saved to
the output_folder.
"""

import cv2
import re
import pandas as pd
from tqdm import tqdm  # This is for the progress bar
import numpy as np
from .settings import settings


# Args:
# dataset_root_path (Path): Path to the dataset root directory. This directory should contain an images directory
# with individual frames, and optionally a labels directory with ground truth labels in YOLO format. If no bbox_path
# is provided, then the function will look for labels and tracks in the dataset root directory.
# output_folder (Path): Path to the output folder for the video file.
# resize (float): If less than 10, ratio by which to resize the frames (e.g., 0.5 for half size). If greater than 10,
# width of the output video.
# bbox_path (Path): Path to the bounding box prediction file (MOT15 format). If provided, the function will use this
# file to generate the video. If not provided, the function will look for labels and tracks in the dataset root
# directory.
def generate_video_with_labels(dataset_root_path, output_folder, resize=1.0, bbox_path=None):
    if dataset_root_path.name in [settings['images_dir'], settings['tracks_dir'], settings['labels_dir']]:
        raise ValueError("Dataset directory should be the dataset root, not images, labels, or tracks directory.")

    run_name = output_folder.name

    # Get bounding boxes
    if not bbox_path:
        all_bboxes = _get_bboxes_from_dataset_root(dataset_root_path)
        output_video_path = output_folder / f"{run_name}_{settings['gt_video_suffix']}"
    elif bbox_path.suffix == '.txt':
        all_bboxes = _get_bboxes_from_txt(bbox_path)
        output_video_path = output_folder / f"{run_name}_{settings['prediction_video_suffix']}"
    else:
        raise ValueError(
            "Bounding box file must be a .txt file. Leave out argument to use dataset ground truth labels and tracks.")

    # Get image files
    image_folder = dataset_root_path / settings['images_dir']
    image_files = sorted([f for f in image_folder.iterdir() if f.suffix in settings['image_file_extensions']])
    if not image_files:
        print(f"No images found in the {image_folder}")
        return

    # Initialize video writer
    first_image_path = image_files[0]
    first_image = cv2.imread(str(first_image_path))
    height, width, _ = first_image.shape

    # Calculate new dimensions based on the resize ratio
    if resize > 10:
        new_width = int(resize)
        new_height = int(height * new_width / width)
    else:
        new_width = int(width * resize)
        new_height = int(height * resize)

    # Set the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_video_path), fourcc, 30, (new_width, new_height))

    # Get track colors and build tuples for cv2
    colors = [(r, g, b) for r, g, b in settings['track_colors']]
    track_colors = {}

    # Text settings
    font_scale = 0.4
    font_height = int(22 * font_scale)  # Just based on default cv2 font height
    text_vertical_margin = 5

    # Iterate over each frame
    for image_file in tqdm(image_files, desc="Processing frames"):
        match = re.search(settings['frame_number_regex'], image_file.name)
        if match:
            frame_number = int(match.group(1))
        else:
            print(f"Could not parse frame number from file: {image_file}")
            continue

        # Get frame and resize
        frame = cv2.imread(str(image_file))
        frame = cv2.resize(frame, (new_width, new_height))

        # Filter bounding boxes for the current frame
        frame_bboxes = all_bboxes[all_bboxes['frame'] == frame_number]

        # Draw bounding boxes and labels
        for _, row in frame_bboxes.iterrows():
            if np.isnan(row['id']):
                continue

            # Get bounding box information from row
            bbox = _get_bbox_from_yolo_coordinates(row['x'], row['y'], row['w'], row['h'], new_width, new_height)
            track_id = int(row['id'])

            # Assign a color to each track ID
            if track_id not in track_colors:
                track_colors[track_id] = colors[len(track_colors) % len(colors)]

            # Get track label location
            track_label_x = bbox['x_top_left']
            track_label_y = bbox['y_top_left'] - text_vertical_margin
            if track_label_y - font_height - text_vertical_margin < 0:
                track_label_y = bbox['y_top_left'] + bbox['h'] + text_vertical_margin + font_height

            # Draw bounding box, center point, and track ID
            cv2.rectangle(frame, (bbox['x_top_left'], bbox['y_top_left']),
                          (bbox['x_bottom_right'], bbox['y_bottom_right']), track_colors[track_id], 1, cv2.LINE_AA)
            cv2.circle(frame, (bbox['center_x'], bbox['center_y']), 1, track_colors[track_id], -1)
            cv2.putText(frame, f'ID: {track_id}', (track_label_x, track_label_y), cv2.FONT_HERSHEY_DUPLEX,
                        font_scale, track_colors[track_id], 1, cv2.LINE_AA)

        # Add frame ID to the lower left corner
        cv2.putText(frame, f'Frame: {image_file.name}', (10, new_height - 10), cv2.FONT_HERSHEY_DUPLEX,
                    font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        # Write the frame to the video
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()
    print(f"Video saved to {output_video_path}")


def _get_bboxes_from_txt(csv_file):
    bboxes = pd.read_csv(csv_file, header=None)
    bboxes.columns = settings['bbox_file_columns']
    return bboxes


def _get_bboxes_from_dataset_root(dataset_root_path):
    labels_dir = dataset_root_path / settings['labels_dir']
    tracks_dir = dataset_root_path / settings['tracks_dir']
    all_bboxes = []

    for label_file in labels_dir.iterdir():
        track_file = tracks_dir / label_file.name
        match = re.search(settings['frame_number_regex'], label_file.name)
        if match:
            frame_number = int(match.group(1))
        else:
            raise ValueError(f"Could not parse frame number from file: {label_file}")

        if track_file.exists():
            if label_file.stat().st_size == 0:
                continue

            bboxes = pd.read_csv(label_file, header=None, sep=' ', index_col=None)
            tracks = pd.read_csv(track_file, header=None, sep=' ', index_col=None)
            bboxes.columns = settings['bbox_file_columns'][1:6]
            bboxes.insert(0, 'frame', frame_number)
            bboxes['id'] = tracks
            all_bboxes.append(bboxes)
        else:
            raise ValueError(f"Label file {label_file} found without corresponding track file {track_file}")

    bboxes_df = pd.concat(all_bboxes, ignore_index=True)
    return bboxes_df


def _get_bbox_from_yolo_coordinates(x, y, w, h, img_width, img_height):
    bbox = {'center_x': x * img_width, 'center_y': y * img_height, 'w': w * img_width, 'h': h * img_height,
        'x_top_left': (x - w / 2) * img_width, 'y_top_left': (y - h / 2) * img_height,
        'x_bottom_right': (x + w / 2) * img_width, 'y_bottom_right': (y + h / 2) * img_height}
    return {k: int(v) for k, v in bbox.items()}

import subprocess
import click
import os
from PIL import Image


def extract_frames(input_video, output_folder):
    """
    Extract frames from INPUT_VIDEO and save them as jpg files in OUTPUT_FOLDER.
    The frames will be named [original_mp4_name]_[frame_number].jpg with zero-padded frame numbers.
    """

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Get the video name without extension
    video_name = os.path.splitext(os.path.basename(input_video))[0]

    # Use FFmpeg with GPU acceleration to extract frames
    ffmpeg_command = [
        'ffmpeg',
        # '-hwaccel', 'cuda',  # Use CUDA for hardware acceleration if available
        '-i', input_video,
        f'{output_folder}/{video_name}_%06d.jpg'
    ]

    subprocess.run(ffmpeg_command)

    click.echo(f"Frames extracted from {input_video} into {output_folder}.")

    # Get the height of one of the extracted frames
    first_frame_path = os.path.join(output_folder, f'{video_name}_000001.jpg')
    with Image.open(first_frame_path) as img:
        image_height = img.height

    return image_height

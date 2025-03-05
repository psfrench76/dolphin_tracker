"""
This script takes the original .jpg frames of a video and the output from dolphin_tracker (.txt file in MOT15 format),
and generates an .mp4 video. The video is saved to the output location specified in --output_folder, and is resized
according to the --resize_ratio option (this doesn't significantly affect processing speed, and is primarily present
to improve file transfer time).
"""

import cv2
import re
import pandas as pd
from tqdm import tqdm
import numpy as np
import yaml
from pathlib import Path
from .settings import settings

# Args:
# image_folder: (Path) Path to the folder containing image frames.
# bbox_file: (Path) Path to the bounding box prediction file (MOT15 format).
# output_folder: (Path) Path to the output folder for the video file.
# resize_ratio: (float) Ratio by which to resize the frames (e.g., 0.5 for half size).
# gt: (bool) Label as ground truth video

def generate_video(image_folder, bbox_file, output_folder, resize_ratio, gt):
    # Read the bounding box file
    bboxes = pd.read_csv(bbox_file, header=None)
    bboxes.columns = settings['bbox_file_columns']

    # Sort images by filename to ensure correct order
    image_files = sorted([f for f in image_folder.iterdir() if f.suffix in settings['image_file_extensions']])
    if not image_files:
        print("No images found in the specified folder.")
        return

    run_name = output_folder.name

    if gt:
        output_video_path = output_folder / f"{run_name}_{settings['gt_video_suffix']}"
    else:
        output_video_path = output_folder / f"{run_name}_{settings['prediction_video_suffix']}"

    # Initialize video writer
    first_image_path = image_files[0]
    first_image = cv2.imread(str(first_image_path))
    height, width, _ = first_image.shape

    # Calculate new dimensions based on the resize ratio
    new_width = int(width * resize_ratio)
    new_height = int(height * resize_ratio)

    # Set the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_video_path), fourcc, 30, (new_width, new_height))

    # Assign colors to track IDs, excluding cyan
    colors = [
        (255, 0, 0), (0, 255, 0), (255, 255, 0),
        (255, 0, 255), (128, 0, 128), (128, 128, 0),
        (128, 0, 0), (0, 128, 0), (0, 128, 128)
    ]
    track_colors = {}

    # Iterate over each frame
    for image_file in tqdm(image_files, desc="Processing frames"):
        match = re.search(settings['frame_number_regex'], image_file.name)
        if match:
            frame_number = int(match.group(1))
        else:
            print(f"Could not parse frame number from file: {image_file}")
            continue

        frame = cv2.imread(str(image_file))

        # Resize the frame
        frame = cv2.resize(frame, (new_width, new_height))

        # Filter bounding boxes for the current frame and resize their coordinates
        frame_bboxes = bboxes[bboxes['frame'] == frame_number]

        # Draw bounding boxes and labels
        for _, row in frame_bboxes.iterrows():
            if np.isnan(row['id']):
                continue

            center_x = int(row['x'] * new_width)
            center_y = int(row['y'] * new_height)
            w = int(row['w'] * new_width)
            h = int(row['h'] * new_height)

            x = int(center_x - w / 2)
            y = int(center_y - h / 2)

            track_id = row['id']

            # Assign a color to each track ID
            if track_id not in track_colors:
                track_colors[track_id] = colors[len(track_colors) % len(colors)]

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), track_colors[track_id], 2)
            cv2.putText(frame, f'ID: {track_id}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, track_colors[track_id], 2)

        # Add frame ID to the lower left corner
        cv2.putText(frame, f'Frame: {image_file.name}', (10, new_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2)

        # Write the frame to the video
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()
    print(f"Video saved to {output_video_path}")

"""
This script takes the original .jpg frames of a video and the output from dolphin_tracker (.txt file in MOT15 format),
and generates an .mp4 video. The video is saved to the output location specified in --output_folder, and is resized
according to the --resize option (this doesn't significantly affect processing speed, and is primarily present
to improve file transfer time).
"""

import cv2
import re
import pandas as pd
from tqdm import tqdm # This is for the progress bar
import numpy as np
import yaml
from pathlib import Path
from .settings import settings

# Args:
# image_folder: (Path) Path to the folder containing image frames.
# bbox_file: (Path) Path to the bounding box prediction file (MOT15 format).
# output_folder: (Path) Path to the output folder for the video file.
# resize: (float) If less than 10, ratio by which to resize the frames (e.g., 0.5 for half size). If greater than 10, width of the output video.
# gt: (bool) Label as ground truth video

def generate_video(image_folder, bbox_file, output_folder, resize, gt):
    image_files = sorted([f for f in image_folder.iterdir() if f.suffix in settings['image_file_extensions']])
    if not image_files:
        print("No images found in the specified folder.")
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

    run_name = output_folder.name
    if gt:
        output_video_path = output_folder / f"{run_name}_{settings['gt_video_suffix']}"
    else:
        output_video_path = output_folder / f"{run_name}_{settings['prediction_video_suffix']}"

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

    # Get bounding boxes from the CSV file
    all_bboxes = _get_bboxes_from_csv(bbox_file)

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
            cv2.rectangle(frame, (bbox['x_top_left'], bbox['y_top_left']), (bbox['x_bottom_right'], bbox['y_bottom_right']), track_colors[track_id], 1, cv2.LINE_AA)
            cv2.circle(frame, (bbox['center_x'], bbox['center_y']), 1, track_colors[track_id], -1)
            cv2.putText(frame, f'ID: {track_id}', (track_label_x, track_label_y), cv2.FONT_HERSHEY_DUPLEX, font_scale, track_colors[track_id], 1, cv2.LINE_AA)

        # Add frame ID to the lower left corner
        cv2.putText(frame, f'Frame: {image_file.name}', (10, new_height - 10), cv2.FONT_HERSHEY_DUPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        # Write the frame to the video
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()
    print(f"Video saved to {output_video_path}")

def _get_bboxes_from_csv(csv_file):
    bboxes = pd.read_csv(csv_file, header=None)
    bboxes.columns = settings['bbox_file_columns']
    return bboxes

def _get_bbox_from_yolo_coordinates(x, y, w, h, img_width, img_height):
    bbox = {
        'center_x': x * img_width,
        'center_y': y * img_height,
        'w': w * img_width,
        'h': h * img_height,
        'x_top_left': (x - w / 2) * img_width,
        'y_top_left': (y - h / 2) * img_height,
        'x_bottom_right': (x + w / 2) * img_width,
        'y_bottom_right': (y + h / 2) * img_height
    }
    return {k: int(v) for k, v in bbox.items()}

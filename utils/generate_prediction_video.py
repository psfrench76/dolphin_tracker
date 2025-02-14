import click
import cv2
import os
import re
import pandas as pd
from tqdm import tqdm
import numpy as np

"""
This script takes the original .jpg frames of a video and the output from dolphin_tracker (.txt file in MOT15 format),
and generates an .mp4 video. The video is saved to the output location specified in --output_folder, and is resized
according to the --resize_ratio option (this doesn't significantly affect processing speed, and is primarily present
to improve file transfer time).
"""


@click.command()
@click.option('--image_folder', required=True, help="Path to the folder containing image frames.")
@click.option('--bbox_file', required=True, help="Path to the bounding box prediction file (MOT15 format).")
@click.option('--output_folder', required=True, help="Path to the output folder for the video file.")
@click.option('--resize_ratio', default=1.0, type=float, help="Ratio by which to resize the frames (e.g., 0.5 for half size).")
@click.option('--gt', is_flag=True, help="Label as ground truth video")
def generate_video(image_folder, bbox_file, output_folder, resize_ratio, gt):
    # Read the bounding box file
    bboxes = pd.read_csv(bbox_file, header=None)
    bboxes.columns = ['frame', 'id', 'x', 'y', 'w', 'h', 'score', 'class', 'visibility']

    # Sort images by filename to ensure correct order
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    if not image_files:
        click.echo("No images found in the specified folder.")
        return

    run_name = os.path.basename(os.path.normpath(output_folder))

    if gt:
        output_video_path = os.path.join(output_folder, f"{run_name}_gt.mp4")
    else:
        output_video_path = os.path.join(output_folder, f"{run_name}_predictions.mp4")

    # Initialize video writer
    first_image_path = os.path.join(image_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, _ = first_image.shape

    # Calculate new dimensions based on the resize ratio
    new_width = int(width * resize_ratio)
    new_height = int(height * resize_ratio)

    # Set the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 30, (new_width, new_height))

    # Assign colors to track IDs, excluding cyan
    colors = [
        (255, 0, 0), (0, 255, 0), (255, 255, 0),
        (255, 0, 255), (128, 0, 128), (128, 128, 0),
        (128, 0, 0), (0, 128, 0), (0, 128, 128)
    ]
    track_colors = {}

    # Regex pattern to extract frame number and base name
    #pattern = r"(.*)_(\d+)(?=[._](jpg|png|jpeg))"
    pattern = r"(\d+)(?=[._](jpg|png|jpeg))"

    # Iterate over each frame
    for image_file in tqdm(image_files, desc="Processing frames"):

        match = re.search(pattern, image_file)
        if match:
            frame_number = int(match.group(1))
        else:
            click.echo(f"Could not parse frame number from file: {image_file}")
            continue

        frame_path = os.path.join(image_folder, image_file)
        frame = cv2.imread(frame_path)

        # Resize the frame
        frame = cv2.resize(frame, (new_width, new_height))

        # Filter bounding boxes for the current frame and resize their coordinates
        frame_bboxes = bboxes[bboxes['frame'] == frame_number]

        # Draw bounding boxes and labels
        for _, row in frame_bboxes.iterrows():
            # Convert normalized coordinates to pixel coordinates for the resized frame
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
        cv2.putText(frame, f'Frame: {frame_number}', (10, new_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Write the frame to the video
        video_writer.write(frame)

    # Release the video writer
    video_writer.release()
    click.echo(f"Video saved to {output_video_path}")


if __name__ == '__main__':
    generate_video()
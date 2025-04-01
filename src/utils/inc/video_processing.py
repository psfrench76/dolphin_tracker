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
import numpy as np
import subprocess
from tqdm import tqdm  # This is for the progress bar
from pathlib import Path
from PIL import Image
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
    output_folder.mkdir(parents=True, exist_ok=True)

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

    if all_bboxes.shape[1] == 6:
        oriented_bbox = False
    elif all_bboxes.shape[1] == 10:
        oriented_bbox = True
    else:
        raise ValueError(f"Bounding boxes do not have the correct number of columns. Found {all_bboxes.shape[1]}, expected 6 or 10.")

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

            if oriented_bbox:
                # Get oriented bounding box information from row
                bbox = _get_obb_bbox_from_points(row['x1'], row['y1'], row['x2'], row['y2'], row['x3'], row['y3'], row['x4'], row['y4'], new_width, new_height)
            else:
                # Get bounding box information from row
                bbox = _get_bbox_from_yolo_coordinates(row['x'], row['y'], row['w'], row['h'], new_width, new_height)

            track_id = int(row['id'])

            # Assign a color to each track ID
            if track_id not in track_colors:
                track_colors[track_id] = colors[len(track_colors) % len(colors)]



            if oriented_bbox:
                # Draw oriented bounding box
                cv2.line(frame, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), track_colors[track_id], 1, cv2.LINE_AA)
                cv2.line(frame, (bbox['x2'], bbox['y2']), (bbox['x3'], bbox['y3']), track_colors[track_id], 1, cv2.LINE_AA)
                cv2.line(frame, (bbox['x3'], bbox['y3']), (bbox['x4'], bbox['y4']), track_colors[track_id], 1, cv2.LINE_AA)
                cv2.line(frame, (bbox['x4'], bbox['y4']), (bbox['x1'], bbox['y1']), track_colors[track_id], 1, cv2.LINE_AA)

                # TODO: Refine this to show orientation explicitly
                track_label_x = min(bbox['x1'], bbox['x2'], bbox['x3'], bbox['x4'])
                track_label_y = min(bbox['y1'], bbox['y2'], bbox['y3'], bbox['y4']) - text_vertical_margin
                if track_label_y - font_height - text_vertical_margin < 0:
                    track_label_y = max(bbox['y1'], bbox['y2'], bbox['y3'], bbox['y4']) + text_vertical_margin + font_height

                center_point = (bbox['center_x'], bbox['center_y'])
                arrow_point = (bbox['orientation_x'], bbox['orientation_y'])

                cv2.arrowedLine(frame, center_point, arrow_point, track_colors[track_id], 1, cv2.LINE_AA)
            else:
                # Draw bounding box, center point, and track ID
                cv2.rectangle(frame, (bbox['x_top_left'], bbox['y_top_left']),
                              (bbox['x_bottom_right'], bbox['y_bottom_right']), track_colors[track_id], 1, cv2.LINE_AA)

                # Get track label location
                track_label_x = bbox['x_top_left']
                track_label_y = bbox['y_top_left'] - text_vertical_margin
                if track_label_y - font_height - text_vertical_margin < 0:
                    track_label_y = bbox['y_top_left'] + bbox['h'] + text_vertical_margin + font_height

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


# Extract frames from INPUT_VIDEO and save them as jpg files in DATASET_ROOT_PATH/images.
# The frames will be named [original_mp4_name]_[frame_number].jpg with zero-padded frame numbers.
def extract_frames(input_video, dataset_root_path):
    # Ensure the output folder exists
    if dataset_root_path.name in [settings['images_dir'], settings['tracks_dir'], settings['labels_dir']]:
        raise ValueError("Dataset directory should be the dataset root, not images, labels, or tracks directory.")

    dataset_root_path = Path(dataset_root_path)
    output_folder = dataset_root_path / settings['images_dir']
    output_folder.mkdir(parents=True, exist_ok=True)

    # Get the video name without extension
    video_name = input_video.stem

    # Use FFmpeg to extract frames
    ffmpeg_command = [
        'ffmpeg',
        '-i', str(input_video),
        str(output_folder / f'{video_name}_%06d.jpg')
    ]

    subprocess.run(ffmpeg_command)

    print(f"Frames extracted from {input_video} into {output_folder}.")

    # Get the height of one of the extracted frames
    first_frame_path = output_folder / f'{video_name}_000001.jpg'
    with Image.open(first_frame_path) as img:
        image_height = img.height

    return image_height


def _get_bboxes_from_txt(csv_file):
    bboxes = pd.read_csv(csv_file, header=None)
    if bboxes.shape[1] == len(settings['bbox_file_columns']):
        bboxes.columns = settings['bbox_file_columns']
    elif bboxes.shape[1] == len(settings['obb_file_columns']):
        bboxes.columns = settings['obb_file_columns']
    else:
        raise ValueError(f"Bounding box file {csv_file} does not have the correct number of columns. Found {bboxes.shape[1]}, expected {len(settings['bbox_file_columns'])} or {len(settings['obb_file_columns'])}.")
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


            if bboxes.shape[1] == 5:
                bboxes.columns = settings['bbox_file_columns'][1:6]
            elif bboxes.shape[1] == 9:
                bboxes.columns = settings['obb_file_columns'][1:10]
            else:
                raise ValueError(
                    f"Bounding box file {label_file} does not have the correct number of columns. Found {bboxes.shape[1]}, expected 5 or 9.")

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

def _get_obb_bbox_from_points(x1, y1, x2, y2, x3, y3, x4, y4, img_width, img_height):
    center_x = (x1 + x2 + x3 + x4) / 4
    center_y = (y1 + y2 + y3 + y4) / 4
    bbox = {'x1': x1 * img_width, 'y1': y1 * img_height, 'x2': x2 * img_width, 'y2': y2 * img_height,
            'x3': x3 * img_width, 'y3': y3 * img_height, 'x4': x4 * img_width, 'y4': y4 * img_height,
            'center_x': center_x * img_width, 'center_y': center_y * img_height,
            'orientation_x': (x2 + x1) / 2 * img_width, 'orientation_y': (y2 + y1) / 2 * img_height}
    return {k: int(v) for k, v in bbox.items()}




if __package__ is None or __package__ == '':
    from settings import settings
else:
    from .settings import settings

from pathlib import Path
import json


def convert_and_save_label(json_file_path, dataset_dir_path, oriented_bbox=False):
    dataset_dir_path = Path(dataset_dir_path)
    label_dir_path = dataset_dir_path / settings['labels_dir']
    track_dir_path = dataset_dir_path / settings['tracks_dir']

    label_dir_path.mkdir(parents=True, exist_ok=True)
    track_dir_path.mkdir(parents=True, exist_ok=True)

    label_file_path = label_dir_path / f"{json_file_path.stem}.txt"
    track_file_path = track_dir_path / f"{json_file_path.stem}.txt"

    labels, tracks = _load_unique_labels(json_file_path)
    _trim_negative_coordinates(labels)

    if oriented_bbox:
        _convert_labels_to_oriented(labels)
        _write_obb_label(labels, label_file_path)
    else:
        _write_label(labels, label_file_path)

    _write_track(tracks, track_file_path)

def _load_unique_labels(json_file_path):
    labels = []
    tracks = []

    with open(json_file_path, 'r') as file:
        data = json.load(file)

    image_width = data['imageWidth']
    image_height = data['imageHeight']

    for shape in data['shapes']:
        if shape['label'] not in settings['dolphin_classes']:
            continue

        # Extract points
        points = shape['points']
        group_id = shape['group_id'] or 0

        if len(points) != 4:
            continue

        # Calculate bounding box
        x_coords = [p[0] for p in points]
        y_coords = [p[1] for p in points]

        x_min = min(x_coords)
        y_min = min(y_coords)
        x_max = max(x_coords)
        y_max = max(y_coords)

        # Normalize coordinates to YOLO format
        x_center = (x_min + x_max) / 2 / image_width
        y_center = (y_min + y_max) / 2 / image_height
        width = (x_max - x_min) / image_width
        height = (y_max - y_min) / image_height

        new_label = (x_center, y_center, width, height)

        if new_label not in labels:
            labels.append((x_center, y_center, width, height))
            tracks.append(group_id)

    return labels, tracks

def _trim_negative_coordinates(labels):
    for i, label in enumerate(labels):
        x_center, y_center, width, height = label

        # The assumption here is that if an x or y center coordinate is negative, the end of the box which is in the frame
        # should stay stationary. This is done by setting the x or y coordinate to 0 and adjusting the width or height
        # by 2 * the negative coordinate -- this pulls the negative edge of the box slightly closer while maintaining
        # the center as close as possible to the original center.

        # Negative coordinates really shouldn't be a thing -- this is a workaround for labelling error and is an edge case

        if x_center < 0:
            x_start = x_center
            x_width = width
            x_width += 2 * x_start
            labels[i] = (0, y_center, x_width, height)

        if y_center < 0:
            y_start = y_center
            y_height = height
            y_height += 2 * y_start
            labels[i] = (x_center, 0, width, y_height)

def _write_label(labels, label_file_path):
    with open(label_file_path, 'w') as out_file:
        for label in labels:
            x_center, y_center, width, height = label
            out_file.write(f"0 {x_center} {y_center} {width} {height}\n")

# This is a placeholder for oriented bounding box labels, which are not yet implemented
def _write_obb_label(labels, label_file_path):
    with open(label_file_path, 'w') as out_file:
        for label in labels:
            pass

# This is a placeholder for oriented bounding box labels, which are not yet implemented
def _convert_labels_to_oriented(labels):
    for i, label in enumerate(labels):
        x_center, y_center, width, height = label
        pass

def _write_track(tracks, track_file_path):
    with open(track_file_path, 'w') as out_file:
        for track in tracks:
            out_file.write(f"{track}\n")
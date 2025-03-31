"""
This file holds the core functionality of the dataset conversion toolset. It is designed to be used as a module
by other scripts, and is not intended to be run directly. See the function comments for descriptions of functionality.
"""
from mpl_toolkits.mplot3d.art3d import line_collection_2d_to_3d

from .settings import settings
from pathlib import Path
import json
import math


# This function takes a single JSON file path and a dataset directory path, and converts the labels in the JSON file
# to the YOLO format and saves them to a file with the same name (but a .txt extension) in the labels directory in the
# dataset directory. It also saves the track IDs to the tracks directory in the dataset directory.

# json_file_path: The path to the JSON file to be converted
# dataset_dir_path: The path to the directory where the labels and tracks directories are located (the dataset root)
# oriented_bbox: A boolean flag indicating whether the labels should be converted to oriented bounding box format.

# Returns a dictionary of statistics about the conversion process, which can be used by calling scripts, aggregated, and
# then passed again to the print_run_stats function to print a summary. See src/utils/copy_and_convert_all_labels.py for
# a usage example.
def convert_and_save_label(json_file_path, dataset_dir_path, oriented_bbox=False):
    frame_stats = {'unique_labels': 0,  # Total unique labels (not frames, some frames have 0 or 2+ labels)
                   'duplicate_labels': 0,  # Total duplicate labels (bounding boxes are identical)
                   'negative_coordinates_trimmed': 0,  # Total labels with negative coordinates trimmed
                   'duplicate_tracks_renumbered': 0,  # Total duplicate tracks renumbered
                   'dolphins_without_keypoints': 0 # Total dolphins without keypoints
                   }
    dataset_dir_path = Path(dataset_dir_path)
    label_dir_path = dataset_dir_path / settings['labels_dir']
    track_dir_path = dataset_dir_path / settings['tracks_dir']

    label_dir_path.mkdir(parents=True, exist_ok=True)
    track_dir_path.mkdir(parents=True, exist_ok=True)

    label_file_path = label_dir_path / f"{json_file_path.stem}.txt"
    track_file_path = track_dir_path / f"{json_file_path.stem}.txt"

    labels, tracks, keypoints, unique_stats = _load_unique_labels(json_file_path, oriented_bbox=oriented_bbox)  # Loads labels and deduplicates them
    #print(labels)
    #print(tracks)
    #print(keypoints)
    #print(json_file_path.stem)
    if oriented_bbox:
        labels = _convert_labels_to_oriented(labels, keypoints)
    trim_stats = _trim_negative_coordinates(labels, oriented_bbox=oriented_bbox)  # Trims labels with negative coordinates
    _increment_all_tracks(tracks)  # Increment all track IDs by 1 -- tracker cannot handle 0s which are endemic
    dedup_stats = _deduplicate_tracks(tracks)  # Deduplicate tracks (different from labels)

    frame_stats.update(unique_stats)
    frame_stats.update(trim_stats)
    frame_stats.update(dedup_stats)

    if oriented_bbox:
        _write_obb_label(labels, label_file_path)
    else:
        _write_label(labels, label_file_path)

    _write_track(tracks, track_file_path)

    return frame_stats


# This is a helper method for scripts which use this module. run_stats should be formatted as a dictionary of
# dictionaries, where the outer dictionary is keyed by the statistic name, the inner dictionary is keyed by the file
# stem of the frame, and the value is the statistic value for that frame. See convert_and_save_label() comments for key
# descriptions.
def print_run_stats(run_stats):
    total_duplicate_tracks = 0
    total_unique_labels = 0
    total_frames = 0
    for key, value in run_stats.items():
        if key == 'unique_labels':
            total_unique_labels = sum(value.values())
            total_frames = len(value)
        elif key == 'duplicate_labels':
            if sum(value.values()) > 0:
                print(f"Total duplicate labels: {sum(value.values())}")
                print(f"Duplicate labels by frame:")
                for k, v in sorted(value.items()):
                    if v > 0:
                        print(f"  {k}: {v}")
        elif key == 'negative_coordinates_trimmed':
            print(f"Total labels with negative coordinates trimmed: {sum(value.values())}")
            if sum(value.values()) > 0:
                print(f"Negative coordinates trimmed by frame:")
                for k, v in sorted(value.items()):
                    if v > 0:
                        print(f"  {k}: {v}")
        elif key == 'duplicate_tracks_renumbered':
            total_duplicate_tracks = sum(value.values())
            print(f"Total duplicate tracks renumbered: {total_duplicate_tracks}")
            if total_duplicate_tracks > 0:
                print(f"Duplicate tracks renumbered by frame:")
                for k, v in sorted(value.items()):
                    if v > 0:
                        print(f"  {k}: {v}")
        elif key == 'dolphins_without_keypoints':
            if sum(value.values()) > 0:
                print(f"Total dolphins without keypoints: {sum(value.values())}")
                print(f"Dolphins without keypoints by frame:")
                for k, v in sorted(value.items()):
                    if v > 0:
                        print(f"  {k}: {v}")

    # ignore frames which only have one label -- they can't have duplicate tracks
    labels_excluding_firsts = total_unique_labels - total_frames
    duplicate_track_ratio = total_duplicate_tracks / labels_excluding_firsts if labels_excluding_firsts != 0 else 0
    print(f"Total frames: {total_frames}")
    print(f"Total unique labels: {total_unique_labels}")
    print(f"Proportion of duplicate tracks: {duplicate_track_ratio:.2%}")
    if duplicate_track_ratio > .15:  # This is an arbitrary threshold -- it's a warning, not an error
        print("WARNING: High proportion of duplicate tracks detected. This may indicate a problem with the labeling. "
              "Consider regenerating track ids for this dataset using the regenerate_tracks.py script.")


# This function takes a track file path and increments all track IDs by the specified amount. This is particularly
# useful for aggregate datasets, which contain labels and tracks from multiple clips. Separating the tracks by clip
# allows for the tracker metrics to distinguish them.
def increment_track_file_by_amount(track_file_path, amount):
    track_file_path = Path(track_file_path)
    tracks = _load_tracks(track_file_path)
    _increment_all_tracks(tracks, amount)
    _write_track(tracks, track_file_path)


# This function takes a label file path and creates a corresponding track file with the same name in the same dataset.
# This is useful for creating background tracks for frames with empty labels.
def create_background_tracks_file(label_file_path):
    label_file_path = Path(label_file_path)
    track_dir_path = label_file_path.parent.parent / settings['tracks_dir']
    track_dir_path.mkdir(exist_ok=True)
    track_file_path = track_dir_path / f"{label_file_path.stem}.txt"
    track_file_path.touch()


# Below this line are private functions which are not intended to be used outside of this module.

# IO functions

def _load_tracks(track_file_path):
    with open(track_file_path, 'r') as track_file:
        track_ids = [int(line.strip()) for line in track_file.readlines()]
        return track_ids


def _write_track(tracks, track_file_path):
    with open(track_file_path, 'w') as out_file:
        for track in tracks:
            out_file.write(f"{track}\n")


def _write_label(labels, label_file_path):
    with open(label_file_path, 'w') as out_file:
        for label in labels:
            x_center, y_center, width, height = label
            out_file.write(f"0 {x_center} {y_center} {width} {height}\n")


# This is a placeholder for oriented bounding box labels, which are not yet implemented
def _write_obb_label(labels, label_file_path):
    with open(label_file_path, 'w') as out_file:
        for label in labels:
            x1, y1, x2, y2, x3, y3, x4, y4 = label
            out_file.write(f"0 {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}\n")
            out_file.write(f"0 {x1} {y1} {x2} {y2} {x3} {y3} {x4} {y4}\n")


# Label load, normalization, conversion, and deduplication
def _load_unique_labels(json_file_path, oriented_bbox=False):
    labels = []
    tracks = []
    heads = {}
    tails = {}
    keypoints = []
    stats = {'unique_labels': 0, 'duplicate_labels': 0, 'dolphins_without_keypoints': 0}

    with open(json_file_path, 'r') as file:
        data = json.load(file)

    image_width = data['imageWidth']
    image_height = data['imageHeight']

    for shape in data['shapes']:
        if shape['label'] in settings['head_classes']:
            group_id = shape['group_id'] or 0
            head_x = shape['points'][0][0] / image_width
            head_y = shape['points'][0][1] / image_height
            heads[group_id] = (head_x, head_y)
        elif shape['label'] in settings['tail_classes']:
            group_id = shape['group_id'] or 0
            tail_x = shape['points'][0][0] / image_width
            tail_y = shape['points'][0][1] / image_height
            tails[group_id] = (tail_x, tail_y)
        elif shape['label'] in settings['dolphin_classes']:
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
                stats['unique_labels'] += 1
            else:
                stats['duplicate_labels'] += 1

    # Reprocess to ensure keypoints and labels are aligned
    if oriented_bbox:
        for i, (label, track) in reversed(list(enumerate(zip(labels, tracks)))):
            if track in heads and track in tails:
                head_x, head_y = heads[track]
                tail_x, tail_y = tails[track]
                keypoints.append((head_x, head_y, tail_x, tail_y))
            else:
                stats['dolphins_without_keypoints'] += 1
                stats['unique_labels'] -= 1
                labels.pop(i)
                tracks.pop(i)

    return labels, tracks, keypoints, stats


# Trim negative coordinates to 0 and adjust width or height to maintain the inner edge of the box
def _trim_negative_coordinates(labels, oriented_bbox=False):
    stats = {'negative_coordinates_trimmed': 0}
    for i, label in enumerate(labels):
        if oriented_bbox:
            x1, y1, x2, y2, x3, y3, x4, y4 = label
            if x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0 or x3 < 0 or y3 < 0 or x4 < 0 or y4 < 0:
                stats['negative_coordinates_trimmed'] += 1
                raise NotImplementedError("Oriented bounding box trimming not yet implemented.")

        else:
            x_center, y_center, width, height = label

            # The assumption here is that if an x or y center coordinate is negative, the end of the box which is in the
            # frame should stay stationary. This is done by setting the x or y coordinate to 0 and adjusting the width or
            # height by 2 * the negative coordinate -- this pulls the negative edge of the box slightly closer while
            # maintaining the center as close as possible to the original center.

            # Negative coordinates really shouldn't be a thing -- this is a workaround for labelling error and is an edge
            # case

            if x_center < 0 or y_center < 0:
                stats['negative_coordinates_trimmed'] += 1

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

    return stats


def _convert_labels_to_oriented(labels, keypoints_list):
    new_labels = []
    for i, (label, keypoints) in enumerate(zip(labels, keypoints_list)):
        x_center, y_center, width, height = label
        head_x, head_y, tail_x, tail_y = keypoints

        # Calculate the angle of the dolphin
        angle = math.atan2(tail_y - head_y, tail_x - head_x)

        # Generate a rectangle containing the full original bounding box

        #TODO - I think Ultralytics does some interpretation of the orientation based on the angle of the box, so
        # the point order may be inconsistent - won't be able to investigate until I have the full visualization pipeline

        # TODO: Filter angles. exactly 0, 90 degrees should just inherit the original bounding box. Other angles should
        #   be clamped.

        #print(f"Index: {i}, Angle: {angle}")

        if angle == 0 or angle == math.pi / 2:
            new_labels.append((x_center - width/2, y_center + height/2,
                                 x_center + width/2, y_center + height/2,
                                 x_center + width/2, y_center - height/2,
                                 x_center - width/2, y_center - height/2))
            continue

        # Normalize to between 0 and 90 degrees for the sake of rectangle generation
        while angle < 0:
            angle += math.pi / 2

        while angle > math.pi / 2:
            angle -= math.pi / 2

        #print(f"Index: {i}, Normalized angle: {angle}")

        # Calculate the 4 lines defining the new bounding box, normalized around origin
        slope1 = math.tan(angle)
        slope2 = 1 / slope1
        intercept1 = slope1 * width/2 + height/2
        intercept2 = math.tan(angle + math.pi / 2) * width/2 - height/2
        line1 = (-slope1, intercept1)
        line2 = (slope2, intercept2)
        line3 = (-slope1, -intercept1)
        line4 = (slope2, -intercept2)

        #print(f"Index: {i}, Lines: {line1}, {line2}, {line3}, {line4}")

        x1, y1 = _line_intersection(line1, line2)
        x2, y2 = _line_intersection(line2, line3)
        x3, y3 = _line_intersection(line3, line4)
        x4, y4 = _line_intersection(line4, line1)

        x1 += x_center
        y1 += y_center
        x2 += x_center
        y2 += y_center
        x3 += x_center
        y3 += y_center
        x4 += x_center
        y4 += y_center

        #print(f"Index: {i}, Points: {x1}, {y1}, {x2}, {y2}, {x3}, {y3}, {x4}, {y4}")

        new_labels.append((x1, y1, x2, y2, x3, y3, x4, y4))

    #print(labels)
    #print(new_labels)

    return new_labels

# Finds the x and y coordinates of the intersection of two lines, given as slope and intercept.
def _line_intersection(line1, line2):
    slope1, intercept1 = line1
    slope2, intercept2 = line2
    x = (intercept2 - intercept1) / (slope1 - slope2)
    y = slope1 * x + intercept1
    return x, y

# Increments a tracks array by a specified amount
def _increment_all_tracks(tracks, amount=1):
    for i in range(len(tracks)):
        tracks[i] += amount


# Deduplicate tracks by renumbering them. Intended as a one-off fix for occasional labeling mistakes. More systemic
# deduplication should be done with regenerate_tracks.py.
def _deduplicate_tracks(tracks):
    stats = {'duplicate_tracks_renumbered': 0}
    seen_tracks = {}

    for i, track in enumerate(tracks):
        seen_tracks[track] = seen_tracks.get(track, 0) + 1
        if seen_tracks[track] > 1:
            tracks[i] = _first_unused_track_id(tracks)
            stats['duplicate_tracks_renumbered'] += 1
            seen_tracks[track] -= 1
            seen_tracks[tracks[i]] = 1
    return stats


# This will return the lowest numbered track ID which is not in the list of tracks. It will interleave numbers which
# are missing rather than defaulting to max() because sometimes duplicates are just a single-frame mistake,
# and it's better as a default to try to guess the correct track ID rather than defaulting to the highest number. But
# this has no knowledge about other frames so it's by nature not perfect.
def _first_unused_track_id(tracks):
    for i in range(min(tracks), min(tracks) + len(
            tracks) + 1):  # +2 because range is exclusive and if the tracks are sequential we want to return the
        # next number
        if i not in tracks:
            return i

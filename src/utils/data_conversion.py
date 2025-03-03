if __package__ is None or __package__ == '':
    from settings import settings
else:
    from .settings import settings

from pathlib import Path
import json


def convert_and_save_label(json_file_path, dataset_dir_path, oriented_bbox=False):
    frame_stats = {
        'unique_labels': 0,
        'duplicate_labels': 0,
        'negative_coordinates_trimmed': 0,
        'duplicate_tracks_renumbered': 0
    }
    dataset_dir_path = Path(dataset_dir_path)
    label_dir_path = dataset_dir_path / settings['labels_dir']
    track_dir_path = dataset_dir_path / settings['tracks_dir']

    label_dir_path.mkdir(parents=True, exist_ok=True)
    track_dir_path.mkdir(parents=True, exist_ok=True)

    label_file_path = label_dir_path / f"{json_file_path.stem}.txt"
    track_file_path = track_dir_path / f"{json_file_path.stem}.txt"

    labels, tracks, unique_stats = _load_unique_labels(json_file_path)
    trim_stats = _trim_negative_coordinates(labels)
    _increment_all_tracks(tracks) # Increment all track IDs by 1 -- tracker cannot handle 0s which are endemic
    dedup_stats = _deduplicate_tracks(tracks)

    frame_stats.update(unique_stats)
    frame_stats.update(trim_stats)
    frame_stats.update(dedup_stats)

    if oriented_bbox:
        _convert_labels_to_oriented(labels)
        _write_obb_label(labels, label_file_path)
    else:
        _write_label(labels, label_file_path)

    _write_track(tracks, track_file_path)

    return frame_stats

# This is a helper method for scripts which use this module. run_stats should be formatted as a dictionary of dictionaries,
# where the outer dictionary is keyed by the statistic name, the inner dictionary is keyed by the file stem of the
# frame, and the value is the statistic value for that frame.
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

    duplicate_track_ratio = total_duplicate_tracks / (total_unique_labels - total_frames) #ignore frames which only have one label
    print(f"Total frames: {total_frames}")
    print(f"Total unique labels: {total_unique_labels}")
    print(f"Proportion of duplicate tracks: {duplicate_track_ratio:.2%}")
    if duplicate_track_ratio > .15: # This is an arbitrary threshold -- it's a warning, not an error
        print("WARNING: High proportion of duplicate tracks detected. This may indicate a problem with the labeling. "
              "Consider regenerating track ids for this dataset using the regenerate_tracks.py script.")

def increment_track_file_by_amount(track_file_path, amount):
    track_file_path = Path(track_file_path)
    tracks = _load_tracks(track_file_path)
    _increment_all_tracks(tracks, amount)
    _write_track(tracks, track_file_path)

def create_background_tracks_file(label_file_path):
    label_file_path = Path(label_file_path)
    track_file_path = Path(str(label_file_path).rsplit(settings['labels_dir'], 1)[0] + settings['tracks_dir'] +
                           str(label_file_path).rsplit(settings['labels_dir'], 1)[1])
    track_file_path.touch()

def _load_tracks(track_file_path):
    with open(track_file_path, 'r') as track_file:
        track_ids = [int(line.strip()) for line in track_file.readlines()]
        return track_ids

def _load_unique_labels(json_file_path):
    labels = []
    tracks = []
    stats = {'unique_labels': 0, 'duplicate_labels': 0}

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
            stats['unique_labels'] += 1
        else:
            stats['duplicate_labels'] += 1


    return labels, tracks, stats

def _trim_negative_coordinates(labels):
    stats = {'negative_coordinates_trimmed': 0}
    for i, label in enumerate(labels):
        x_center, y_center, width, height = label

        # The assumption here is that if an x or y center coordinate is negative, the end of the box which is in the frame
        # should stay stationary. This is done by setting the x or y coordinate to 0 and adjusting the width or height
        # by 2 * the negative coordinate -- this pulls the negative edge of the box slightly closer while maintaining
        # the center as close as possible to the original center.

        # Negative coordinates really shouldn't be a thing -- this is a workaround for labelling error and is an edge case

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

def _increment_all_tracks(tracks, amount=1):
    for i in range(len(tracks)):
        tracks[i] += amount

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

# This will return the lowest numbered track ID which is not in the list of tracks. It will interleave numbers which are missing
# rather than defaulting to max() because sometimes duplicates are just a single-frame mistake, and it's better as a
# default to try to guess the correct track ID rather than defaulting to the highest number. But this has no knowledge
# about other frames so it's by nature not perfect.
def _first_unused_track_id(tracks):
    for i in range(1, len(tracks) + 2): # +2 because range is exclusive and if the tracks are sequential we want to return the next number
        if i not in tracks:
            return i

def _write_track(tracks, track_file_path):
    with open(track_file_path, 'w') as out_file:
        for track in tracks:
            out_file.write(f"{track}\n")
"""
This script regenerates track files based on label files. It takes a labels directory and a tracks directory.
For any empty label files, it creates an empty track file if it is missing. It then checks that there are the same number of files
in each directory and raises an error if not, indicating which files are missing in which directory.
It iterates through the labels and tracks files in parallel, updating the track files based on the labels.

Buyer beware! This script is a blunt force instrument. It uses the hungarian algorithm (linear_sum_assignment) to match labels
to tracks based on the distance between the centers of the bounding boxes. This is not guaranteed to be correct, and may
result in incorrect track assignments.

It is only suggested that you use this script when you KNOW that the ground-truth label ids are not correct, and you want
to get somewhere closer to the truth. I highly recommend creating a new copy of your dataset and testing to see if it's better.

Usage: regenerate_tracks.py <labels_dir> <tracks_dir>
"""
import argparse
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment

def load_labels(label_file_path):
    labels = []
    with open(label_file_path, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) == 5:
                x_center, y_center, width, height = map(float, parts[1:])
                labels.append((x_center, y_center, width, height))
    return labels

def load_tracks(track_file_path):
    tracks = []
    with open(track_file_path, 'r') as file:
        for line in file:
            tracks.append(int(line.strip()))
    return tracks

def save_tracks(track_file_path, tracks):
    with open(track_file_path, 'w') as file:
        for track in tracks:
            file.write(f"{track}\n")

def regenerate_tracks(labels_dir, tracks_dir):
    labels_path = Path(labels_dir)
    tracks_path = Path(tracks_dir)

    # Create empty track files for empty label files if they are missing
    for label_file in labels_path.glob('*.txt'):
        if label_file.stat().st_size == 0:
            track_file = tracks_path / label_file.name
            if not track_file.exists():
                track_file.touch()

    # Glob the label and track files after creating missing track files
    label_files = sorted(labels_path.glob('*.txt'))
    track_files = sorted(tracks_path.glob('*.txt'))

    # Check that there are the same number of files in each directory
    if len(label_files) != len(track_files):
        label_file_names = {file.name for file in label_files}
        track_file_names = {file.name for file in track_files}
        missing_in_labels = track_file_names - label_file_names
        missing_in_tracks = label_file_names - track_file_names
        raise ValueError(f"Mismatch in number of files. Missing in labels: {missing_in_labels}. Missing in tracks: {missing_in_tracks}")

    previous_labels = []
    previous_tracks = []
    global_track_id = 1

    for label_file, track_file in zip(label_files, track_files):
        labels = load_labels(label_file)
        if not labels:
            save_tracks(track_file, [])
            continue

        if not previous_labels:
            tracks = list(range(global_track_id, global_track_id + len(labels)))
            global_track_id += len(labels)
        else:
            tracks = [None] * len(labels)
            distances = np.array([[np.linalg.norm(np.array(current_label[:2]) - np.array(prev_label[:2]))
                                   for prev_label in previous_labels]
                                  for current_label in labels])

            row_ind, col_ind = linear_sum_assignment(distances)
            for row, col in zip(row_ind, col_ind):
                tracks[row] = previous_tracks[col]

            for i, track in enumerate(tracks):
                if track is None:
                    tracks[i] = global_track_id
                    global_track_id += 1

        save_tracks(track_file, tracks)
        previous_labels = labels
        previous_tracks = tracks
    print("Tracks regenerated successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Regenerate track files based on label files.")
    parser.add_argument('labels_dir', type=str, help='Path to the labels directory')
    parser.add_argument('tracks_dir', type=str, help='Path to the tracks directory')
    args = parser.parse_args()

    regenerate_tracks(args.labels_dir, args.tracks_dir)

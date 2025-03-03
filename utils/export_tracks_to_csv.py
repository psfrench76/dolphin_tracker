import csv
import argparse
from pathlib import Path

def load_tracks(track_file_path):
    tracks = []
    with open(track_file_path, 'r') as file:
        for line in file:
            tracks.append(int(line.strip()))
    return tracks

def export_tracks_to_csv(tracks_dir, csv_path):
    tracks_path = Path(tracks_dir)
    track_files = sorted(tracks_path.glob('*.txt'))

    # Create the directory if it does not exist
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        for track_file in track_files:
            frame_id = track_file.stem
            tracks = load_tracks(track_file)
            csv_writer.writerow([frame_id] + tracks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export track files to a CSV file.")
    parser.add_argument('tracks_dir', type=str, help='Path to the tracks directory')
    parser.add_argument('csv_path', type=str, help='Path to the output CSV file')
    args = parser.parse_args()

    export_tracks_to_csv(args.tracks_dir, args.csv_path)
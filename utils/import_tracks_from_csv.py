import csv
import argparse
from pathlib import Path

def save_tracks(track_file_path, tracks):
    with open(track_file_path, 'w') as file:
        for track in tracks:
            file.write(f"{track}\n")

def import_tracks_from_csv(tracks_dir, csv_path):
    tracks_path = Path(tracks_dir)

    with open(csv_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        for row in csv_reader:
            frame_id = row[0]
            tracks = [int(track) for track in row[1:] if track]
            track_file = tracks_path / f"{frame_id}.txt"
            save_tracks(track_file, tracks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Import track data from a CSV file.")
    parser.add_argument('tracks_dir', type=str, help='Path to the tracks directory')
    parser.add_argument('csv_path', type=str, help='Path to the input CSV file')
    args = parser.parse_args()

    import_tracks_from_csv(args.tracks_dir, args.csv_path)
import json
from pathlib import Path
from collections import defaultdict
import argparse
from inc.settings import settings

"""
This script takes an input directory of .json or .txt files (from the labelme converted format or other formats) 
and returns the number of frames with each number of dolphins.
"""


def count_dolphins_per_frame(dataset_root, print_empty=False):
    if not dataset_root.is_dir():
        print(f"Error: {dataset_root} is not a valid directory.")
        return

    if dataset_root.name in [settings['images_dir'], settings['tracks_dir'], settings['labels_dir'], settings['orientations_dir']]:
        raise ValueError("Destination directory should be the dataset root, not images, labels, tracks, or orientations directory.")

    labels_dir = dataset_root / settings['labels_dir']

    dolphin_counts = defaultdict(int)

    # Iterate over all files in the directory
    for file_path in Path(labels_dir).rglob('*'):
        if file_path.suffix == '.txt':
            # Open and read the TXT file
            if print_empty:
                if file_path.stat().st_size == 0:
                    print(f"Empty frame: {file_path}")
                    continue
            with file_path.open('r') as file:
                lines = file.readlines()

            # Use a set to track unique lines
            unique_lines = set(line.strip() for line in lines if line.strip())

            # Count the number of unique lines (dolphins) in the current frame
            num_dolphins = len(unique_lines)
            dolphin_counts[num_dolphins] += 1

    for num_dolphins, num_frames in sorted(dolphin_counts.items(), reverse=True):
        print(f"{num_dolphins} dolphins: {num_frames} frames")


def main():
    parser = argparse.ArgumentParser(description="Count the number of dolphins per frame in a dataset.")
    parser.add_argument('dataset_root', type=Path, help='Path to the input directory containing .json or .txt files')
    parser.add_argument('--print_empty ', action='store_true', help='Print frames with no dolphins')
    args = parser.parse_args()

    count_dolphins_per_frame(args.dataset_root, args.print_empty)


if __name__ == '__main__':
    main()

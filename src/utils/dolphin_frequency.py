import json
from pathlib import Path
from collections import defaultdict
import argparse
from inc.settings import settings

"""
This script takes an input directory of .json or .txt files (from the labelme converted format or other formats) 
and returns the number of frames with each number of dolphins.
"""


def count_dolphins_per_frame(dataset_root):
    if not dataset_root.is_dir():
        print(f"Error: {dataset_root} is not a valid directory.")
        return

    dolphin_counts = defaultdict(int)

    # Iterate over all files in the directory
    for file_path in Path(dataset_root).rglob('*'):
        if file_path.suffix == '.json':
            # Open and load the JSON file
            with file_path.open('r') as file:
                data = json.load(file)

            # Use a set to track unique group_ids for dolphins
            dolphin_group_ids = set()

            # Iterate over shapes and find group_ids for "Dolphin" shapes
            for shape in data.get('shapes', []):
                if shape.get('label') in ["Dolphin", "dolphin", "box", "rectangle"]:
                    dolphin_group_ids.add(shape.get('group_id'))

            # Count the number of dolphins in the current frame
            num_dolphins = len(dolphin_group_ids)
            dolphin_counts[num_dolphins] += 1

        elif file_path.suffix == '.txt':
            # Open and read the TXT file
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
    args = parser.parse_args()

    count_dolphins_per_frame(args.dataset_root)


if __name__ == '__main__':
    main()

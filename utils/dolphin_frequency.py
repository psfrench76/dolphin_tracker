import os
import json
import sys
from collections import defaultdict

"""
This script takes an input directory of .json files (from the labelme converted format) and returns the number of frames
with each number of dolphins, similar to a histogram.
"""

def count_dolphins_per_frame(directory):
    dolphin_counts = defaultdict(int)

    # Iterate over all files in the directory
    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.json'):
                file_path = os.path.join(root, filename)

                # Open and load the JSON file
                with open(file_path, 'r') as file:
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

    return dolphin_counts


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_directory>")
        sys.exit(1)

    input_directory = sys.argv[1]
    dolphin_counts = count_dolphins_per_frame(input_directory)

    for num_dolphins, num_frames in sorted(dolphin_counts.items(), reverse=True):
        print(f"{num_dolphins} dolphins: {num_frames} frames")
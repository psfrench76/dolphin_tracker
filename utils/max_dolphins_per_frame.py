import os
import json
import sys

"""
This script takes an input directory of .json files (from the labelme converted format) and returns the maximum number
of dolphins in any of the label files. As this can vary by frame this is useful to find out what the maximum number
of dolphins in a given video is.
"""

def find_max_dolphins_per_frame(directory):
    max_dolphins = 0

    # Iterate over all files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)

            # Open and load the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)

            # Use a set to track unique group_ids for dolphins
            dolphin_group_ids = set()

            # Iterate over shapes and find group_ids for "Dolphin" shapes
            for shape in data.get('shapes', []):
                if shape.get('label') == 'Dolphin':
                    dolphin_group_ids.add(shape.get('group_id'))

            # Update the maximum number of dolphins found in a single frame
            max_dolphins = max(max_dolphins, len(dolphin_group_ids))

    return max_dolphins


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_directory>")
        sys.exit(1)

    input_directory = sys.argv[1]
    max_dolphins = find_max_dolphins_per_frame(input_directory)
    print(f"The maximum number of dolphins per frame is: {max_dolphins}")
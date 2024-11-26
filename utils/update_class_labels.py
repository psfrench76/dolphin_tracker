import os
import json
import argparse

"""
Processes JSON annotation files to update dolphin labels.

This function iterates over all JSON files in the specified directory,
filters out shapes that do not have exactly four points (i.e., non-rectangles),
and updates the labels of the remaining shapes. If the `include_group_ids` flag
is set, the labels are formatted as "Dolphin_X", where X is the group ID.
Otherwise, all labels are simply set to "dolphin".

Args:
    raw_jsons_directory (str): Path to the directory containing the JSON annotation files.
    include_group_ids (bool): Whether to include group IDs in the dolphin labels.
"""

def process_annotations(raw_jsons_directory, include_group_ids):
    for filename in os.listdir(raw_jsons_directory):
        if filename.endswith('.json'):
            filepath = os.path.join(raw_jsons_directory, filename)

            with open(filepath, 'r') as file:
                data = json.load(file)

            data["shapes"] = [shape for shape in data["shapes"] if len(shape["points"]) == 4]
            for shape in data['shapes']:
                if include_group_ids and 'group_id' in shape and shape['group_id'] is not None:
                    shape['label'] = f"Dolphin_{shape['group_id']}"
                else:
                    shape['label'] = "dolphin"

            with open(filepath, 'w') as file:
                json.dump(data, file, indent=4)

def main():
    parser = argparse.ArgumentParser(description='Process JSON annotations to update dolphin labels.')
    parser.add_argument('jsons_directory', type=str, help='Path to the directory containing the JSON annotation files.')
    parser.add_argument('--groupids', action='store_true', help='Include group IDs in the dolphin labels.')

    args = parser.parse_args()

    process_annotations(args.jsons_directory, args.groupids)

if __name__ == "__main__":
    main()
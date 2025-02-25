import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse

"""
This script combines .jpg files from a Roboflow dataset and .json files from an original source
into an output folder. It creates two subdirectories in the output folder: 'images' for the
.jpg files and 'labels' for the .json files. The script only copies files if both a .jpg and
a matching .json file exist, where a match is determined by the base filename (excluding extensions).

Optionally, the script can copy original .jpg files from the original source instead of the Roboflow
dataset. This is controlled by the --copyoriginaljpgs flag. The copying process is multi-threaded
for efficiency. At the end of the run, the script reports how many examples from the Roboflow dataset
were skipped and the reasons for skipping (e.g., missing source files).

Example usage:
python utils/reconstruct_labelme.py --roboflow_dataset ../data/roboflow_dolphines_v8/train/ --output_folder 
../data/rf_dolphines_v8_labelme/train/ --original_source ../data/original_source/collated/ --copyoriginaljpgs
"""


def index_files(original_source):
    # Create indices for JSON and JPEG files for quick lookup
    json_index = {}
    jpg_index = {}
    for root, _, files in os.walk(original_source):
        for file in files:
            basename, ext = os.path.splitext(file)
            if ext == '.json':
                json_index[basename] = os.path.join(root, file)
            elif ext == '.jpg':
                jpg_index[basename] = os.path.join(root, file)
    return json_index, jpg_index


def copy_file(src_path, dest_path):
    # Copy a single file from src_path to dest_path
    shutil.copy2(src_path, dest_path)


def main(roboflow_dataset, output_folder, original_source, copy_original_jpgs):
    # Create output directories if they don't exist
    images_output = os.path.join(output_folder, 'images')
    labels_output = os.path.join(output_folder, 'labels')
    os.makedirs(images_output, exist_ok=True)
    os.makedirs(labels_output, exist_ok=True)

    # Index JSON and JPEG files
    json_index, jpg_index = index_files(original_source)

    # List of files to copy
    files_to_copy = []
    skipped_files = []

    # Walk through the roboflow_dataset directory to find JPG files
    images_input = os.path.join(roboflow_dataset, 'images')
    for root, _, files in os.walk(images_input):
        for file in files:
            if file.endswith('.jpg'):
                # Replace '_jpg' with '.jpg' and truncate at the first instance
                truncated_name = file.split('_jpg', 1)[0]

                # Check if corresponding JSON exists
                if truncated_name in json_index:
                    # Determine source and destination paths
                    jpg_src_path = (
                        jpg_index[truncated_name]
                        if copy_original_jpgs and truncated_name in jpg_index
                        else os.path.join(root, file)
                    )
                    jpg_dest_path = os.path.join(images_output, truncated_name + '.jpg')
                    json_src_path = json_index[truncated_name]
                    json_dest_path = os.path.join(labels_output, truncated_name + '.json')

                    if jpg_src_path:
                        # Append to the list of files to copy
                        files_to_copy.append((jpg_src_path, jpg_dest_path))
                        files_to_copy.append((json_src_path, json_dest_path))
                    else:
                        skipped_files.append((truncated_name, "Missing original JPEG"))
                else:
                    skipped_files.append((truncated_name, "Missing JSON file"))

    # Copy files using multi-threading
    total_files = len(files_to_copy)
    copied_files = 0

    with ThreadPoolExecutor() as executor:
        future_to_path = {executor.submit(copy_file, src, dest): dest for src, dest in files_to_copy}
        for future in as_completed(future_to_path):
            copied_files += 1
            print(f"{copied_files}/{total_files} files copied", end='\r')

    print("\nCopying completed.")
    print(f"Skipped {len(skipped_files)} files.")
    for name, reason in skipped_files:
        print(f"Skipped {name}: {reason}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine JPEG and JSON files into an output folder.")
    parser.add_argument('--roboflow_dataset', required=True, help='Path to the Roboflow dataset')
    parser.add_argument('--output_folder', required=True, help='Path to the output folder')
    parser.add_argument('--original_source', required=True, help='Path to the original JSON and JPEG source')
    parser.add_argument('--copyoriginaljpgs', action='store_true', help='Copy original JPEGs instead of Roboflow JPEGs')

    args = parser.parse_args()

    main(args.roboflow_dataset, args.output_folder, args.original_source, args.copyoriginaljpgs)

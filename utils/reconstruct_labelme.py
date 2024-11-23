import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

"""
This script combines .jpg files from a Roboflow dataset and .json files from a JSON source
into an output folder. It creates two subdirectories in the output folder: 'images' for the
.jpg files and 'labels' for the .json files. The script only copies files if both a .jpg and
a matching .json file exist, where a match is determined by the base filename (excluding extensions).
The copying process is multi-threaded for efficiency.

Example usage:
python reconstruct_labelme.py roboflow_dolphines_v8/test/ rf_dolphines_v8_labelme/test/ original_source/collated/json/
"""


def index_json_files(json_source):
    # Create an index of JSON files for quick lookup
    json_index = {}
    for root, _, files in os.walk(json_source):
        for file in files:
            if file.endswith('.json'):
                basename = os.path.splitext(file)[0]
                json_index[basename] = os.path.join(root, file)
    return json_index


def copy_file(src_path, dest_path):
    # Copy a single file from src_path to dest_path
    shutil.copy2(src_path, dest_path)


def main(roboflow_dataset, output_folder, json_source):
    # Create output directories if they don't exist
    images_output = os.path.join(output_folder, 'images')
    labels_output = os.path.join(output_folder, 'labels')
    os.makedirs(images_output, exist_ok=True)
    os.makedirs(labels_output, exist_ok=True)

    # Index JSON files
    json_index = index_json_files(json_source)

    # List of files to copy
    files_to_copy = []

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
                    jpg_src_path = os.path.join(root, file)
                    jpg_dest_path = os.path.join(images_output, truncated_name + '.jpg')
                    json_src_path = json_index[truncated_name]
                    json_dest_path = os.path.join(labels_output, truncated_name + '.json')

                    # Append to the list of files to copy
                    files_to_copy.append((jpg_src_path, jpg_dest_path))
                    files_to_copy.append((json_src_path, json_dest_path))

    # Copy files using multi-threading
    total_files = len(files_to_copy)
    copied_files = 0

    with ThreadPoolExecutor() as executor:
        future_to_path = {executor.submit(copy_file, src, dest): dest for src, dest in files_to_copy}
        for future in as_completed(future_to_path):
            copied_files += 1
            print(f"{copied_files}/{total_files} files copied", end='\r')

    print("\nCopying completed.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <roboflow_dataset> <output_folder> <json_source>")
        sys.exit(1)

    roboflow_dataset = sys.argv[1]
    output_folder = sys.argv[2]
    json_source = sys.argv[3]

    main(roboflow_dataset, output_folder, json_source)
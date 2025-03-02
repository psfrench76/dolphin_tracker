import os
import shutil
import argparse
from pathlib import Path

def index_files(directory):
    """
    Index all files in the directory and its subdirectories.
    """
    file_index = {}
    for file in Path(directory).rglob('*'):
        if file.is_file():
            file_index[file.name] = file
    return file_index

def copy_files(input_label_folder, original_source_folder):
    """
    Copy files from the original source to the input label folder if they exist.
    """
    input_label_path = Path(input_label_folder)
    original_source_path = Path(original_source_folder)

    # Index files in the original source folder
    original_files = index_files(original_source_path)

    copied_files_count = 0
    empty_files_not_found_count = 0
    non_empty_files_not_found_count = 0

    for label_file in input_label_path.iterdir():
        if label_file.is_file():
            if label_file.name in original_files:
                shutil.copy(original_files[label_file.name], label_file)
                copied_files_count += 1
            else:
                if label_file.stat().st_size == 0:
                    empty_files_not_found_count += 1
                else:
                    non_empty_files_not_found_count += 1

    print(f"Total files copied: {copied_files_count}")
    print(f"Total empty files not found: {empty_files_not_found_count}")
    print(f"Total non-empty files not found: {non_empty_files_not_found_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Copy updated labels from the original source to the input label folder.")
    parser.add_argument('input_label_folder', type=str, help='Path to the input label folder')
    parser.add_argument('original_source_folder', type=str, help='Path to the original source folder')
    args = parser.parse_args()

    copy_files(args.input_label_folder, args.original_source_folder)
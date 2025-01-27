import os
import shutil
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

"""
This script walks through a file tree resembling the Box repository originally provided by Labrinto, and
copies frames and labels based on video name and file type in a structured way to make things simpler to find.

It extracts the video name from the file name, and the file type by the file extension.

This is identical to collate_datafiles.py, except that it is threaded for faster performance.

Usage: python collate_datafiles_threaded.py <input_path> <output_path>
"""

def sanitize_filename(filename):
    # Remove version numbers in parentheses e.g., "(1)"
    filename = re.sub(r'\(\d+\)', '', filename)
    return filename

def extract_video_name(filename, directory_name):
    # Find the video name by splitting the filename at the last underscore
    if '_' in filename:
        return filename.rsplit('_', 1)[0]
    else:
        return directory_name

def copy_file(src_path, dest_path):
    # Copy a single file from src_path to dest_path
    shutil.copy2(src_path, dest_path)
    return dest_path

def copy_files(input_path, output_path):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Walk through the input directory
    files_to_copy = []
    for root, _, files in os.walk(input_path):
        directory_name = os.path.basename(root)
        for file in files:
            if not file.endswith('.zip'):  # Ignore .zip files
                sanitized_file = sanitize_filename(file)
                file_extension = os.path.splitext(sanitized_file)[1].lstrip('.').lower()  # Get file extension
                if file_extension:  # Check if file extension exists
                    video_name = extract_video_name(sanitized_file, directory_name)
                    ext_dir = os.path.join(output_path, file_extension)
                    video_dir = os.path.join(ext_dir, video_name)
                    if not os.path.exists(video_dir):
                        os.makedirs(video_dir)
                    src_path = os.path.join(root, file)
                    dest_path = os.path.join(video_dir, sanitized_file)
                    files_to_copy.append((src_path, dest_path))

    total_files = len(files_to_copy)
    copied_files = 0

    # Use ThreadPoolExecutor to copy files in parallel
    with ThreadPoolExecutor() as executor:
        future_to_path = {executor.submit(copy_file, src, dest): dest for src, dest in files_to_copy}
        for future in as_completed(future_to_path):
            copied_files += 1
            print(f"{copied_files}/{total_files} files copied", end='\r')

    print("\nCopying completed.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_path> <output_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    copy_files(input_path, output_path)
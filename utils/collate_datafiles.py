import os
import shutil
import re
import sys

"""
This script walks through a file tree resembling the Box repository originally provided by Labrinto, and
copies frames and labels based on video name and file type in a structured way to make things simpler to find.

It extracts the video name from the file name, and the file type by the file extension.

Usage: python collate_datafiles.py <input_path> <output_path>
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
                    files_to_copy.append((root, file, sanitized_file, file_extension, video_name))

    total_files = len(files_to_copy)
    copied_files = 0

    # Copy files and show progress
    for root, file, sanitized_file, file_extension, video_name in files_to_copy:
        # Create directories based on file extensions and video names
        ext_dir = os.path.join(output_path, file_extension)
        video_dir = os.path.join(ext_dir, video_name)
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)

        # Source and destination paths
        src_path = os.path.join(root, file)
        dest_path = os.path.join(video_dir, sanitized_file)

        # Copy file
        shutil.copy2(src_path, dest_path)
        copied_files += 1

        # Display progress
        print(f"{copied_files}/{total_files} files copied", end='\r')

    print("\nCopying completed.")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_path> <output_path>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    copy_files(input_path, output_path)
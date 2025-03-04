"""
This script copies image and label files from an input directory to an output directory
based on a specified frame number range. The range is inclusive of the start frame and
exclusive of the end frame.

Usage: copy_images_and_labels_in_frame_range.py <input_folder> <output_folder> <start_frame> <end_frame>
"""

import shutil
import re
import argparse
from pathlib import Path


# Copy files from input_folder to output_folder whose frame numbers are between start_frame and end_frame.
# The range is inclusive of the start_frame and exclusive of the end_frame.
def copy_frame_range(input_folder, output_folder, start_frame, end_frame):
    output_folder.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r'_(\d+)\.')

    copied_files_count = 0

    for filename in input_folder.iterdir():
        match = pattern.search(filename.name)
        if match:
            frame_number = int(match.group(1))
            if start_frame <= frame_number < end_frame:
                dst_path = output_folder / filename.name
                shutil.copy(filename, dst_path)
                copied_files_count += 1

    print(f"Total files copied: {copied_files_count}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Copy files from input folder to output folder based on frame number range.")
    parser.add_argument('input_folder', type=Path, help='Path to the input folder')
    parser.add_argument('output_folder', type=Path, help='Path to the output folder')
    parser.add_argument('start_frame', type=int, help='Start frame number (inclusive)')
    parser.add_argument('end_frame', type=int, help='End frame number (exclusive)')
    args = parser.parse_args()

    copy_frame_range(args.input_folder, args.output_folder, args.start_frame, args.end_frame)

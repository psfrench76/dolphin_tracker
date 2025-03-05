"""
This script copies image, label, and track files from an input dataset to an output dataset
based on a specified frame number range. The range is inclusive of the start frame and
exclusive of the end frame. If skip_images_without_labels is True, images without labels will be skipped, otherwise all
images will be copied.

Usage: copy_images_and_labels_in_frame_range.py <source_dataset_root> <dest_dataset_root> <start_frame>
        <end_frame> [--skip_images_without_labels]
"""

import shutil
import re
import argparse
from pathlib import Path
from inc.settings import settings


# Copy images, labels, and tracks from source_dataset_root to dest_dataset_root whose frame numbers are between
# start_frame and end_frame. The range is inclusive of the start_frame and exclusive of the end_frame.
def _copy_frame_range(source_dataset_root, dest_dataset_root, start_frame, end_frame, skip_images_without_labels):
    source_dataset_root_path = Path(source_dataset_root)
    dest_dataset_root_path = Path(dest_dataset_root)

    if dest_dataset_root_path.name in [settings['images_dir'], settings['tracks_dir'], settings['labels_dir']]:
        raise ValueError("Destination directory should be the dataset root, not images, labels, or tracks directory.")

    source_images_path = source_dataset_root_path / settings['images_dir']
    source_labels_path = source_dataset_root_path / settings['labels_dir']
    source_tracks_path = source_dataset_root_path / settings['tracks_dir']

    dest_images_path = dest_dataset_root_path / settings['images_dir']
    dest_labels_path = dest_dataset_root_path / settings['labels_dir']
    dest_tracks_path = dest_dataset_root_path / settings['tracks_dir']

    dest_images_path.mkdir(parents=True, exist_ok=True)
    dest_labels_path.mkdir(parents=True, exist_ok=True)
    dest_tracks_path.mkdir(parents=True, exist_ok=True)

    frame_id_pattern = re.compile(r'_(\d+)\.')

    copied_frames_count = 0
    copied_images_count = 0
    copied_labels_count = 0
    copied_tracks_count = 0
    skipped_images = []

    for source_image_file in source_images_path.iterdir():
        match = frame_id_pattern.search(source_image_file.name)
        if match:
            frame_number = int(match.group(1))
            if start_frame <= frame_number < end_frame:
                source_label_file = source_labels_path / (source_image_file.stem + '.txt')
                source_track_file = source_tracks_path / (source_image_file.stem + '.txt')
                if not skip_images_without_labels or source_label_file.exists():
                    dest_image_file = dest_images_path / source_image_file.name
                    shutil.copy(source_image_file, dest_image_file)
                    if source_label_file.exists():
                        dest_label_file = dest_labels_path / source_label_file.name
                        shutil.copy(source_label_file, dest_label_file)
                        copied_labels_count += 1
                    if source_track_file.exists():
                        dest_track_file = dest_tracks_path / source_track_file.name
                        shutil.copy(source_track_file, dest_track_file)
                        copied_tracks_count += 1
                    copied_frames_count += 1
                    copied_images_count += 1
                else:
                    skipped_images.append(source_image_file.name)

    print(f"Total frames copied: {copied_frames_count}")
    print(f"Total images copied: {copied_images_count}")
    print(f"Total labels copied: {copied_labels_count}")
    print(f"Total tracks copied: {copied_tracks_count}")
    if skipped_images:
        print(f"WARNING: Skipped {len(skipped_images)} images without labels:")
        for image_file_name in skipped_images:
            print(image_file_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Copy files from input folder to output folder based on frame number range.")
    parser.add_argument('source_dataset_root', type=Path, help='Path to the input folder')
    parser.add_argument('dest_dataset_root', type=Path, help='Path to the output folder')
    parser.add_argument('start_frame', type=int, help='Start frame number (inclusive)')
    parser.add_argument('end_frame', type=int, help='End frame number (exclusive)')
    parser.add_argument('--skip_images_without_labels', action='store_true', help='Skip images without labels')
    args = parser.parse_args()

    _copy_frame_range(args.source_dataset_root, args.dest_dataset_root, args.start_frame, args.end_frame,
                      args.skip_images_without_labels)

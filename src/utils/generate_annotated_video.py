"""
This script takes the original .jpg frames of a video and the output from dolphin_tracker (.txt file in MOT15 format),
and generates an .mp4 video. The video is saved to the output location specified in --output_folder, and is resized
according to the --resize option (this doesn't significantly affect processing speed, and is primarily present
to improve file transfer time).
"""

import argparse
from pathlib import Path
from inc.video_generation import generate_video

def main():
    parser = argparse.ArgumentParser(description="Generate a video from image frames and bounding box predictions.")
    parser.add_argument('--image_folder', required=True, type=Path, help="Path to the folder containing image frames.")
    parser.add_argument('--bbox_file', required=True, type=Path, help="Path to the bounding box prediction file (MOT15 format).")
    parser.add_argument('--output_folder', required=True, type=Path, help="Path to the output folder for the video file.")
    parser.add_argument('--resize', default=1.0, type=float, help="If less than 10, ratio by which to resize the frames (e.g., 0.5 for half size). If greater than 10, width of the output video.")
    parser.add_argument('--gt', action='store_true', help="Label as ground truth video")
    args = parser.parse_args()

    generate_video(args.image_folder, args.bbox_file, args.output_folder, args.resize, args.gt)

if __name__ == '__main__':
    main()
from inc.video_processing import extract_frames
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from a video.")
    parser.add_argument('input_video', type=Path, help="Path to the video file.")
    parser.add_argument('output_folder', type=Path, help="Path to the output folder for the frames.")
    args = parser.parse_args()
    extract_frames(args.input_video, args.output_folder)

if __name__ == '__main__':
    main()

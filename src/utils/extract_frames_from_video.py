from inc.video_processing import extract_frames
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Extract frames from a video.")
    parser.add_argument('input_video', type=Path, help="Path to the video file.")
    parser.add_argument('dataset_root_path', type=Path, help="Path to the dataset root for the frames. The frames will be saved in DATASET_ROOT_PATH/images.")
    args = parser.parse_args()
    extract_frames(args.input_video, args.dataset_root_path)

if __name__ == '__main__':
    main()

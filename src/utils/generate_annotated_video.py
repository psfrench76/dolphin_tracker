"""
This script takes the original .jpg frames of a video and the output from dolphin_tracker (.txt file in MOT15 format),
and generates an .mp4 video. The video is saved to the output location specified in --output_folder, and is resized
according to the --resize option (this doesn't significantly affect processing speed, and is primarily present
to improve file transfer time).
"""

import argparse
from pathlib import Path
from inc.video_processing import generate_video_with_labels


def main():
    parser = argparse.ArgumentParser(
        description="Generate a video from image frames and bounding box predictions or ground truth.")
    parser.add_argument('--dataset_root_path', '-d', required=True, type=Path, help="Path to the dataset root folder")
    parser.add_argument('--output_folder', '-o', required=True, type=Path,
                        help="Path to the output folder for the video file.")
    parser.add_argument('--resize', '-r', default=1.0, type=float,
                        help="If less than 10, ratio by which to resize the frames (e.g., 0.5 for half size). If "
                             "greater than 10, width of the output video.")
    parser.add_argument('--bbox_path', '-bb', type=Path,
                        help="Path to the bounding box prediction file (MOT15 format).")
    parser.add_argument('--orientations_outfile', '-oo', type=Path,
                        help="Path to the orientations output file (optional).")
    parser.add_argument('--start_frame', '-sf', type=int, default=0, )
    parser.add_argument('--end_frame', '-ef', type=int, default=-1,
                        help="End frame for the video. Default is -1 (all frames).")
    parser.add_argument('--ignore_bbox', '-ib', action='store_true', )
    parser.add_argument('--researcher_csv', '-rc', type=Path,
                        help="Path to the researcher CSV file (optional).")
    parser.add_argument('--researcher_csv_col', '-rcc', type=str,
                        help="Name of angle column to use in the researcher CSV file (optional). Default is 'MovingAvgAngle_deg'.")
    args = parser.parse_args()

    generate_video_with_labels(args.dataset_root_path, args.output_folder, args.resize, args.bbox_path,
                               args.orientations_outfile, sf=args.start_frame, ef=args.end_frame,
                               ignore_bbox=args.ignore_bbox, researcher_csv=args.researcher_csv,
                               csv_angle_column=args.researcher_csv_col)


if __name__ == '__main__':
    main()

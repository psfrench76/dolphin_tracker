#!/usr/bin/env python3

import sys
import os
import argparse
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.track import run_tracking_and_evaluation
from utils.generate_prediction_video import generate_video
from utils.parse_video import extract_frames

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_data', help="Path to the video or dataset.")
    parser.add_argument('--prediction_video', '-pv', action='store_true', help="Generate a video of the tracking results.")
    parser.add_argument('--ground_truth_video', '-gv', action='store_true', help="Generate a video of the ground truth labels.")
    parser.add_argument('--break_apart', '-b', action='store_true', help="Break video into individual frames and store in a dataset directory.")

    args = parser.parse_args()
    run_args = vars(args)

    break_apart = args.break_apart
    input_data = args.input_data

    summary_log = "--------------------------------------------\n\n"

    with open('cfg/settings.yaml') as f:
        settings = yaml.safe_load(f)

    model = settings['default_detector']
    tracker = settings['default_tracker']
    resize_ratio = settings['default_video_resize_ratio']
    # TODO: make resize ratio an argument, in pixels instead of ratio (generate_video should distinguish)

    if os.path.isfile(input_data):
        if args.ground_truth_video:
            raise ValueError(f"Ground truth videos not supported with video input. Please use a dataset directory "
                             f"containing individual frames in an '{settings['images_dir']}' directory and labels in a "
                             f"'{settings['labels_dir']}' directory.")

        if args.prediction_video:
            break_apart = True
            print("\n Will break video into individual frames so that prediction video can be generated\n")
            summary_log += "Broke video into individual frames so that prediction video could be generated\n"

        if break_apart:
            print("\nBreaking video into individual frames...\n")
            input_name = os.path.basename(input_data).split('.')[0]
            dataset = os.path.join(settings['dataset_dir'], settings['extracted_frames_dir'], input_name)
            images_dir = os.path.join(str(dataset), settings['images_dir'])
            extract_frames(input_data, images_dir)
            summary_log += f"Video broken into individual frames and stored in {images_dir}\n"
        else:
            # TODO - implement mp4/mov input for tracker
            input_name = os.path.basename(input_data).split('.')[0]
            dataset = input_data
            raise ValueError("Video input for tracker not implemented yet")

    elif os.path.isdir(input_data):
        if break_apart:
            print("Ignoring --break_apart flag as input is a directory.")
            summary_log += f"Ignored --break_apart flag as input {input_data} is a directory.\n"

        if args.ground_truth_video and not os.path.isdir(os.path.join(input_data, settings['labels_dir'])):
            raise FileNotFoundError(f"Dataset folder must contain a '{settings['labels_dir']}' directory in order "
                                    f"to generate a ground truth video.")

        if not os.path.isdir(os.path.join(input_data, settings['images_dir'])):
            raise FileNotFoundError(f"Dataset folder must contain an '{settings['images_dir']}' directory")

        head, tail = os.path.split(input_data)
        if not tail:
            head, tail = os.path.split(head)
        input_name = tail

        if tail in settings['dataset_split_dirs']:
            _, tail = os.path.split(head)
            input_name = tail + '_' + input_name

        dataset = input_data
    else:
        raise ValueError("Input data must be a file or a directory.")

    output_dir = os.path.join(settings['output_dir'], settings['tracker_output_dir'], input_name)
    os.makedirs(output_dir, exist_ok=True)
    arg_outfile = os.path.join(str(output_dir), settings['args_file'])

    run_args['output_dir'] = output_dir
    run_args['model'] = model
    run_args['tracker'] = tracker
    run_args['dataset'] = dataset

    with open(arg_outfile, "w") as f:
        yaml.dump(run_args, f)

    output_filename = f"{input_name}_{settings['researcher_output_suffix']}"

    # To be removed in later version
    os.chdir('..')
    output_dir = output_dir[3:]
    model = model[3:]
    dataset = dataset[3:]
    tracker = os.path.join('dolphin_tracker', tracker)
    # End filesystem reconciliation

    run_tracking_and_evaluation(dataset, model, output_dir, tracker)

    # To be removed in later version
    os.chdir('dolphin_tracker')
    output_dir = os.path.join('..', output_dir)
    model = os.path.join('..', model)
    dataset = os.path.join('..', dataset)
    tracker = os.path.join('..', tracker)
    # End filesystem reconciliation

    summary_log +=  f"Tracking and evaluation complete. CSV results can be found in "\
                    f"{os.path.join(output_dir, output_filename)}\n"

    print(f"\nTracking and evaluation complete.\n")

    if args.prediction_video:
        print(f"Generating prediction video...")

        bbox_file = os.path.join(output_dir, f"{input_name}_{settings['results_file_suffix']}")
        images_folder = os.path.join(dataset, settings['images_dir'])
        generate_video(images_folder, bbox_file, output_dir, resize_ratio, False)

        video_filename = f"{input_name}_{settings['prediction_video_suffix']}"
        video_path = os.path.join(output_dir, video_filename)
        print(f"Prediction video output written to {video_path}")
        summary_log += f"Prediction video output written to {video_path}\n"

    if args.ground_truth_video:
        print(f"Generating ground truth video...")

        gt_bbox_file = os.path.join(output_dir, f"{input_name}_{settings['gt_file_suffix']}")
        images_folder = os.path.join(dataset, settings['images_dir'])
        generate_video(images_folder, gt_bbox_file, output_dir, resize_ratio, True)

        video_filename = f"{input_name}_{settings['gt_video_suffix']}"
        video_path = os.path.join(output_dir, video_filename)
        print(f"Ground truth video output written to {video_path}")
        summary_log += f"Ground truth video output written to {video_path}\n"

    print(summary_log)

if __name__ == '__main__':
    main()
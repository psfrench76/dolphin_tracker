#!/usr/bin/env python3

import argparse
import yaml
from pathlib import Path
from src.track import run_tracking_and_evaluation
from utils.generate_prediction_video import generate_video
from utils.parse_video import extract_frames
from src.utils.settings import settings, storage_path, project_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_data', help="Path to the video or dataset.")
    parser.add_argument('--prediction_video', '-pv', action='store_true', help="Generate a video of the tracking results.")
    parser.add_argument('--ground_truth_video', '-gv', action='store_true', help="Generate a video of the ground truth labels.")
    parser.add_argument('--break_apart', '-b', action='store_true', help="Break video into individual frames and store in a dataset directory.")

    args = parser.parse_args()
    run_args = vars(args)

    break_apart = args.break_apart
    input_path = Path(args.input_data)
    input_name = input_path.stem

    summary_log = "--------------------------------------------\n\n"

    model_path = storage_path(settings['default_detector'])
    tracker_path = project_path(settings['default_tracker'])

    resize_ratio = settings['default_video_resize_ratio']
    image_dir_name = settings['images_dir']
    label_dir_name = settings['labels_dir']
    # TODO: make resize ratio an argument, in pixels instead of ratio (generate_video should distinguish)

    if input_path.is_file():
        if args.ground_truth_video:
            raise ValueError(f"Ground truth videos not supported with video input. Please use a dataset directory "
                             f"containing individual frames in an '{image_dir_name}' directory and labels in a "
                             f"'{label_dir_name}' directory.")

        if args.prediction_video:
            break_apart = True
            print("\n Will break video into individual frames so that prediction video can be generated\n")
            summary_log += "Broke video into individual frames so that prediction video could be generated\n"

        if break_apart:
            print("\nBreaking video into individual frames...\n")
            dataset_path = storage_path(f'data/extracted/{input_name}')
            image_dir_path = dataset_path / image_dir_name
            extract_frames(input_path, image_dir_path)
            summary_log += f"Video broken into individual frames and stored in {image_dir_path}\n"
        else:
            # TODO - implement mp4/mov input for tracker
            dataset_path = input_path
            raise ValueError("Video input for tracker not implemented yet")

    elif input_path.is_dir():
        if break_apart:
            print("Ignoring --break_apart flag as input is a directory.")
            summary_log += f"Ignored --break_apart flag as input {input_path} is a directory.\n"

        if args.ground_truth_video and not (input_path / label_dir_name).is_dir():
            raise FileNotFoundError(f"Dataset folder must contain a '{label_dir_name}' directory in order "
                                    f"to generate a ground truth video.")

        if not (input_path / image_dir_name).is_dir():
            raise FileNotFoundError(f"Dataset folder must contain an '{image_dir_name}' directory")



        if input_name in settings['dataset_split_dirs']:
            input_name = input_path.parent.name + '_' + input_name

        dataset_path = input_path
    else:
        raise ValueError("Input data must be a file or a directory.")

    output_dir_path = storage_path(f"output/tracker/{input_name}")
    output_dir_path.mkdir(parents=True, exist_ok=True)
    arg_outfile_path = output_dir_path / settings['args_file']

    run_args['output_dir'] = str(output_dir_path)
    run_args['model'] = str(model_path)
    run_args['tracker'] = str(tracker_path)
    run_args['dataset'] = str(dataset_path)

    with open(arg_outfile_path, "w") as f:
        yaml.dump(run_args, f)

    output_filename = f"{input_name}_{settings['researcher_output_suffix']}"

    run_tracking_and_evaluation(dataset_path, model_path, output_dir_path, tracker_path)

    summary_log +=  f"Tracking and evaluation complete. CSV results can be found in "\
                    f"{output_dir_path / output_filename}\n"

    print(f"\nTracking and evaluation complete.\n")

    if args.prediction_video:
        print(f"Generating prediction video...")

        results_file_path = output_dir_path / f"{input_name}_{settings['results_file_suffix']}"
        image_dir_path = dataset_path / image_dir_name
        generate_video(image_dir_path, results_file_path, output_dir_path, resize_ratio, False)

        video_filename = f"{input_name}_{settings['prediction_video_suffix']}"
        video_path = output_dir_path / video_filename
        print(f"Prediction video output written to {video_path}")
        summary_log += f"Prediction video output written to {video_path}\n"

    if args.ground_truth_video:
        print(f"Generating ground truth video...")

        gt_file_path = output_dir_path / f"{input_name}_{settings['gt_file_suffix']}"
        image_dir_path = dataset_path / image_dir_name
        generate_video(image_dir_path, gt_file_path, output_dir_path, resize_ratio, True)

        video_filename = f"{input_name}_{settings['gt_video_suffix']}"
        video_path = output_dir_path / video_filename
        print(f"Ground truth video output written to {video_path}")
        summary_log += f"Ground truth video output written to {video_path}\n"

    print(summary_log)

if __name__ == '__main__':
    main()
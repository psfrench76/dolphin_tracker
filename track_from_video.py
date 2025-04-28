#!/usr/bin/env python3

import argparse
import yaml
from pathlib import Path
from src.track import run_tracking_and_evaluation
from src.utils.inc.video_processing import generate_video_with_labels, extract_frames
from src.utils.inc.settings import settings, storage_path, project_path, get_device_and_workers
from src.utils.inc.reporting import DataAccumulator
from src.utils.inc.orientation_network import OrientationResNet
from src.utils.inc.orientation_dataset import DolphinOrientationDataset
from torch.utils.data import DataLoader
import torch
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_data', type=Path, help="Path to the video or dataset.")
    parser.add_argument('--prediction_video', '-pv', action='store_true',
                        help="Generate a video of the tracking results.")
    parser.add_argument('--ground_truth_video', '-gv', action='store_true',
                        help="Generate a video of the ground truth labels.")
    parser.add_argument('--break_apart', '-b', action='store_true',
                        help="Break video into individual frames and store in a dataset directory.")
    parser.add_argument('--srt', '-srt', type=Path, help="Path to an SRT file corresponding to the video input.")
    parser.add_argument('--model', '-m', type=Path, help="Path to the model (weights) file.")
    parser.add_argument('--orientation_model', '-om', type=Path,
                        help="Path to the orientation model (weights) file.")
    parser.add_argument('--tracker', '-t', type=Path, help="Path to the tracker file.")
    parser.add_argument('--drone_config', '-dc', type=str,
                        help=f"Drone configuration file, in the {settings['drone_profile_dir']} directory")
    parser.add_argument('--resize_ratio', '-rr', type=float,
                        help="Resize ratio for the video. Only one of -rr, -rw should be used.")
    parser.add_argument('--resize_width', '-rw', type=int,
                        help="Width in pixels to resize the video to. Only one of -rr, -rw should be used.")
    parser.add_argument('--calibration', '-c', type=float,
                        help="Calibration factor for converting pixels to meters in output.csv. This will be "
                             "multiplied to the final value as determined by the drone configuration. Overrides the "
                             "calibration factor in the drone configuraton.")
    parser.add_argument('--altitude', '-a', type=float,
                        help="Manual altitude in meters for the full video, for converting pixels to meters in "
                             "output.csv. This can take the place of an SRT file if one is missing.")
    parser.add_argument('--output', '-o', type=Path,
                        help="Path to the output directory. Default is output/tracker.")

    args = parser.parse_args()
    run_args = vars(args)

    break_apart = args.break_apart
    input_path = args.input_data
    srt_path = args.srt
    input_name = input_path.stem
    summary_log = SummaryLog()
    image_dir_name = settings['images_dir']
    label_dir_name = settings['labels_dir']

    # Argument parsing and checking
    model_path = args.model or storage_path(settings['default_detector'])
    tracker_path = args.tracker or project_path(settings['default_tracker'])

    if input_path.is_file():
        if args.ground_truth_video:
            raise ValueError(f"Ground truth videos not supported with video input. Please use a dataset directory "
                             f"containing individual frames in an '{image_dir_name}' directory and labels in a "
                             f"'{label_dir_name}' directory.")

        if args.prediction_video:
            break_apart = True
            print("\n Will break video into individual frames so that prediction video can be generated\n")
            summary_log.add("Broke video into individual frames so that prediction video could be generated")

        if break_apart:
            print("\nBreaking video into individual frames...\n")
            dataset_path = storage_path(f"{settings['data_dir']}/{settings['extracted_frames_dir']}/{input_name}")
            extract_frames(input_path, dataset_path)
            summary_log.add(f"Video broken into individual frames and stored in {dataset_path / image_dir_name}")
        else:
            # TODO - implement mp4/mov input for tracker.
            dataset_path = input_path
            raise ValueError("Video input for tracker not implemented yet")

    elif input_path.is_dir():
        if break_apart:
            print("Ignoring --break_apart flag as input is a directory.")
            summary_log.add(f"Ignored --break_apart flag as input {input_path} is a directory.")

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

    if srt_path:
        if not srt_path.is_file():
            raise FileNotFoundError(f"SRT file not found at {srt_path}")

        print(f"Using SRT file at {srt_path}")

    if args.output:
        output_dir_path = args.output
        output_file_basename = args.output.stem
    else:
        output_dir_path = storage_path(f"{settings['output_dir']}/{settings['tracker_output_dir']}/{input_name}")
        output_file_basename = input_name

    output_dir_path.mkdir(parents=True, exist_ok=True)
    arg_outfile_path = output_dir_path / settings['args_file']

    run_args['output_dir'] = str(output_dir_path)
    run_args['model'] = str(model_path)
    run_args['tracker'] = str(tracker_path)
    run_args['dataset'] = str(dataset_path)
    run_args['srt'] = str(srt_path)

    with open(arg_outfile_path, "w") as f:
        yaml.dump(run_args, f)

    output_filename = f"{output_file_basename}_{settings['researcher_output_suffix']}"
    output_file_path = output_dir_path / output_filename
    tracker_results_path = output_dir_path / f"{output_file_basename}_{settings['results_file_suffix']}"
    images_index_file = output_dir_path / f"{output_file_basename}_{settings['images_index_suffix']}"

    researcher_data_accumulator = DataAccumulator(bbox_type='xyxy', units='pct')

    run_tracking_and_evaluation(dataset_path, model_path, output_dir_path, tracker_path, srt_path=srt_path,
                                drone_profile=args.drone_config, calibration=args.calibration,
                                manual_altitude=args.altitude, researcher_data_accumulator=researcher_data_accumulator)

    drone_profile = args.drone_config or settings['default_drone_profile']

    if srt_path is not None or args.altitude is not None:
        if srt_path is not None:
            researcher_data_accumulator.load_srt_altitudes(srt_path)
        if args.altitude is not None:
            researcher_data_accumulator.load_manual_altitudes(args.altitude)
        researcher_data_accumulator.add_gsd_column(drone_profile, args.calibration)
        researcher_data_accumulator.add_conversion_columns('m')

    researcher_data_accumulator.add_individual_count_column()
    researcher_data_accumulator.add_distances_columns()


    summary_log.add(f"Tracking and evaluation complete.")

    print(f"\nTracking and evaluation complete.\n")

    print(f"\nStarting dolphin orientation prediction...\n")
    device, num_workers = get_device_and_workers()
    orientation_model_path = args.orientation_model or storage_path(settings['default_orientation_model'])
    orientations_outfile_path = output_dir_path / f"{output_file_basename}_{settings['orientations_results_suffix']}"

    print(f"Predicting on dataset {dataset_path}. Loading model weights from {orientation_model_path}")

    dataset = DolphinOrientationDataset(dataset_root_dir=dataset_path, annotations=tracker_results_path, images_index_file=images_index_file)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=num_workers)

    model = OrientationResNet()
    model.load_state_dict(torch.load(orientation_model_path, map_location=device, weights_only=True))
    model.set_device(device)

    pred_df = model.predict(dataloader, orientations_outfile_path)

    researcher_data_accumulator.load_orientations(orientations_outfile_path)
    researcher_data_accumulator.to_csv(output_file_path, ignore_columns=['Confidence'])

    print(f"Final results saved to {output_file_path}")
    summary_log.add(f"Final results saved to {output_file_path}")

    # The resize argument in generate_video_with_labels can be either a ratio or a pixel width
    if args.resize_ratio:
        resize = args.resize_ratio
    elif args.resize_width:
        resize = args.resize_width
    else:
        resize = settings['default_video_resize_ratio']

    if args.prediction_video:
        print(f"Generating prediction video...")

        results_file_path = output_dir_path / f"{output_file_basename}_{settings['results_file_suffix']}"
        generate_video_with_labels(dataset_path, output_dir_path, resize, results_file_path, orientations_outfile=orientations_outfile_path)

        video_filename = f"{output_file_basename}_{settings['prediction_video_suffix']}"
        video_path = output_dir_path / video_filename
        print(f"Prediction video output written to {video_path}")
        summary_log.add(f"Prediction video output written to {video_path}")

    if args.ground_truth_video:
        print(f"Generating ground truth video...")

        generate_video_with_labels(dataset_path, output_dir_path, resize)

        video_filename = f"{output_file_basename}_{settings['gt_video_suffix']}"
        video_path = output_dir_path / video_filename
        print(f"Ground truth video output written to {video_path}")
        summary_log.add(f"Ground truth video output written to {video_path}")

    print(summary_log)


class SummaryLog:

    def __init__(self):
        self.log = "\n--------------------------------------------\n\n"

    def add(self, message):
        self.log += message + "\n"

    def __str__(self):
        return self.log


if __name__ == '__main__':
    main()

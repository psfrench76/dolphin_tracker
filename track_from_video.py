#!/usr/bin/env python3

import sys
import os
import argparse
import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.track import run_tracking_and_evaluation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_data', help="Path to the video or dataset.")
    parser.add_argument('--generate_video', '-g', action='store_true', help="Generate a video of the tracking results.")
    args = parser.parse_args()
    run_args = vars(args)

    with open('cfg/settings.yaml') as f:
        settings = yaml.safe_load(f)

    model = settings['default_detector']
    tracker = settings['default_tracker']

    if os.path.isfile(args.input_data):
        # TODO
        output_name = os.path.basename(args.input_data).split('.')[0]
        raise ValueError("File conversion not implemented yet")
    elif os.path.isdir(args.input_data):
        head, tail = os.path.split(args.input_data)
        if not tail:
            head, tail = os.path.split(head)

        output_name = tail

        if tail in ['train', 'test', 'val', 'valid']:
            _, tail = os.path.split(head)
            output_name = tail + '_' + output_name

        dataset = args.input_data
    else:
        raise ValueError("Input data must be a file or a directory.")

    output_dir = os.path.join(settings['output_dir'], os.getlogin(), output_name)
    os.makedirs(output_dir, exist_ok=True)
    arg_outfile = os.path.join(str(output_dir), settings['args_file'])

    run_args['output_dir'] = output_dir
    run_args['model'] = model
    run_args['tracker'] = tracker
    run_args['dataset'] = dataset

    with open(arg_outfile, "w") as f:
        yaml.dump(run_args, f)

    # To be removed in later version
    os.chdir('..')
    output_dir = output_dir[3:]
    model = model[3:]
    dataset = dataset[3:]
    tracker = os.path.join('dolphin_tracker', tracker)
    # End filesystem reconciliation

    run_tracking_and_evaluation(dataset, model, output_dir, tracker)

    print(f"\n--------------------------------------------\n"
          f"Tracking and evaluation complete. Results can be found in "
          f"{os.path.join(output_dir, settings['researcher_output_suffix'])}\n")

    if args.generate_video:
        print(f"Generating video...")

        # TODO

        run_name = os.path.basename(os.path.normpath(output_dir))
        video_filename = f"{run_name}_predictions.mp4"
        video_path = os.path.join(output_dir, video_filename)
        print(f"Video output written to {video_path}")

if __name__ == '__main__':
    main()
import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from utils.inc.orientation_network import OrientationResNet
from utils.inc.orientation_dataset import DolphinOrientationDataset
from utils.inc.settings import set_seed, settings, get_device_and_workers
from torch.utils.data import DataLoader
from utils.inc.reporting import OrientationMetrics

import psutil
import torch

def main():
    parser = argparse.ArgumentParser(description="Evaluate orientations against ground truth.")
    parser.add_argument('--dataset', '-d', type=Path, required=True, help="Path to the dataset root directory.")
    parser.add_argument('--output_folder', '-o', type=Path, required=True, help="Path to the output folder.")
    parser.add_argument('--weights', '-w', type=Path, required=True, help="Path to the model weights file.")
    parser.add_argument('--augment', '-a', action='store_true', help="Use data augmentation.")
    parser.add_argument('--imgsz', '-sz', type=int, help="Image size for the model.")
    parser.add_argument('--num_workers', '-nw', type=int, default=None, help="Number of workers for data loading.")
    parser.add_argument('--filter_angles', '-fa', action='store_true', help="Evaluate using angle filtering and averaging. Not compatible with augment.")
    parser.add_argument('--neighbor_window', '-nbw', type=int,
                        help="Number of neighbors to use for filtering angles. Default is 120.")
    parser.add_argument('--angle_window', '-aw', type=int,
                        help="Angle window to use for filtering angles. Default is 25.")
    parser.add_argument('--angle_threshold', '-at', type=float,
                        help="Threshold to use for filtering angles. Default is 0.6.")
    parser.add_argument('--moving_avg_window', '-mw', type=int,
                        help="Window size for moving average when averaging orientations. Default is 20 frames. Note that this is calculated after 180-degree filtering. Passing a value of 1 is equivalent to not using a moving average.")
    args = parser.parse_args()

    dataset_dir = args.dataset
    output_folder = args.output_folder
    weights = args.weights
    outfile_path = output_folder / f"{output_folder.name}_{settings['orientations_results_suffix']}"
    imgsz = args.imgsz if args.imgsz else 244

    output_folder.mkdir(parents=True, exist_ok=True)

    set_seed(0)

    device, num_workers = get_device_and_workers(split=False)
    if args.num_workers is not None:
        num_workers = args.num_workers
        print(f"Overriding num_workers to {num_workers}.")

    dataloader_args = {
        'batch_size': 256,
        'num_workers': num_workers,
        'pin_memory': True,
        'shuffle': True,
        'prefetch_factor': 4,
        #'persistent_workers': True,
    }

    print(f"Predicting on dataset {dataset_dir}. Loading model weights from {weights}")

    if args.filter_angles and args.augment:
        raise ValueError("Data augmentation is not compatible with angle filtering and averaging.")

    dataset = DolphinOrientationDataset(dataset_root_dir=dataset_dir, augment=args.augment, imgsz=imgsz)
    dataloader = DataLoader(dataset, **dataloader_args)

    model = OrientationResNet(weights=weights, device=device)
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.set_device(device)

    om = model.evaluate(dataloader, outfile_path, filter_angles=args.filter_angles,
                        neighbor_window=args.neighbor_window, angle_window=args.angle_window,
                        angle_threshold=args.angle_threshold, moving_avg_window=args.moving_avg_window)

    metrics_file = output_folder / f"{output_folder.name}_{settings['orientations_metrics_suffix']}"
    line_results_file = output_folder / f"{output_folder.name}_{settings['orientations_line_results_suffix']}"

    om.write_results(metrics_file)
    om.write_line_results(line_results_file)
    om.print_results()

    print(f"Metrics saved to {metrics_file}")
    print(f"Line-by-line results saved to {line_results_file}")

if __name__ == "__main__":
    main()
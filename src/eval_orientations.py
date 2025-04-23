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
    args = parser.parse_args()

    dataset_dir = args.dataset
    output_folder = args.output_folder
    weights = args.weights
    outfile_path = output_folder / f"{output_folder.name}_{settings['orientations_results_suffix']}"
    imgsz = args.imgsz if args.imgsz else 244

    output_folder.mkdir(parents=True, exist_ok=True)

    set_seed(0)

    device, num_workers = get_device_and_workers(split=False)

    print(f"Predicting on dataset {dataset_dir}. Loading model weights from {weights}")

    dataset = DolphinOrientationDataset(dataset_root_dir=dataset_dir, augment=args.augment, imgsz=imgsz)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=num_workers)

    model = OrientationResNet()
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.set_device(device)

    all_outputs, all_indices, all_tracks = model.predict(dataloader)
    all_filenames = [str(dataset.get_image_path(idx).stem) for idx in all_indices]

    print(f"Predictions complete. Saving to {outfile_path}")

    # Create a DataFrame
    data = {
        'dataloader_index': all_indices,
        'filename': all_filenames,
        'object_id': all_tracks,
    }
    other_df = pd.DataFrame(data)

    pred_df = model.write_outputs(all_outputs, other_df, outfile_path)
    print(f"Final angles saved to {outfile_path}")

    # Evaluate the predictions
    gt_df = model.get_ground_truth(dataset)

    om = OrientationMetrics(pred_df, gt_df)
    om.calculate_metrics()

    metrics_file = output_folder / f"{output_folder.name}_{settings['orientations_metrics_suffix']}"
    line_results_file = output_folder / f"{output_folder.name}_{settings['orientations_line_results_suffix']}"

    om.write_results(metrics_file)
    om.write_line_results(line_results_file)
    om.print_results()

    print(f"Metrics saved to {metrics_file}")
    print(f"Line-by-line results saved to {line_results_file}")

if __name__ == "__main__":
    main()
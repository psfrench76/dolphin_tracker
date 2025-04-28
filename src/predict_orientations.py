import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from utils.inc.orientation_network import OrientationResNet
from utils.inc.orientation_dataset import DolphinOrientationDataset
from utils.inc.settings import set_seed, settings, get_device_and_workers
from torch.utils.data import DataLoader

import psutil
import torch

def main():
    parser = argparse.ArgumentParser(description="Predict orientations.")
    parser.add_argument('--dataset', '-d', type=Path, required=True, help="Path to the dataset root directory.")
    parser.add_argument('--output_folder', '-o', type=Path, required=True, help="Path to the output folder.")
    parser.add_argument('--weights', '-w', type=Path, required=True, help="Path to the model weights file.")
    parser.add_argument('--tracking_results', '-tr', type=Path, help="Path to the tracking results file (optional).")
    parser.add_argument('--images_index_file', '-ii', type=Path, help="Path to the images index file (optional).")
    args = parser.parse_args()

    dataset_dir = args.dataset
    output_folder = args.output_folder
    weights = args.weights
    tracking_results = args.tracking_results
    images_index_file = args.images_index_file
    outfile_path = output_folder / f"{output_folder.name}_{settings['orientations_results_suffix']}"

    output_folder.mkdir(parents=True, exist_ok=True)

    set_seed(0)

    device, num_workers = get_device_and_workers()

    print(f"Predicting on dataset {dataset_dir}. Loading model weights from {weights}")

    dataset = DolphinOrientationDataset(dataset_root_dir=dataset_dir, annotations=tracking_results, images_index_file=images_index_file)
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

    model.write_outputs(all_outputs, outfile_path, other_df)
    print(f"Final angles saved to {outfile_path}")

if __name__ == "__main__":
    main()
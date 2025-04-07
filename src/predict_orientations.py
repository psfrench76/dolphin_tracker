import argparse
from pathlib import Path
import pandas as pd
from tqdm import tqdm

from utils.inc.orientation_network import OrientationResNet
from utils.inc.orientation_dataset import DolphinOrientationDataset
from utils.inc.settings import set_seed, settings
from torch.utils.data import DataLoader

import psutil
import torch
from torchvision import transforms

def main():
    parser = argparse.ArgumentParser(description="Predict orientations.")
    parser.add_argument('--dataset', '-d', type=Path, required=True, help="Path to the dataset root directory.")
    parser.add_argument('--output_folder', '-o', type=Path, required=True, help="Path to the output folder.")
    parser.add_argument('--weights', '-w', type=Path, required=True, help="Path to the model weights file.")
    args = parser.parse_args()

    dataset_dir = args.dataset
    output_folder = args.output_folder
    weights = args.weights
    outfile_path = output_folder / f"{output_folder.name}_{settings['orientations_results_suffix']}"

    output_folder.mkdir(parents=True, exist_ok=True)

    set_seed(0)

    # TODO - this (cuda and workers) is (almost) the same as in train.py. Let's modularize this stuff.
    #   Note that the UL module uses an array of numbers though. When modularizing make sure that mutli-GPU isn't broken for UL.
    num_cores = len(psutil.Process().cpu_affinity())

    # Set the number of workers to the recommended value
    num_workers = int(min(16, num_cores) / 2)

    print(f"Number of available CPU cores: {num_cores}")
    print(f"Setting number of workers to: {num_workers} (divided by 2 for train/val split)")

    # Check for CUDA availability
    if torch.cuda.is_available():
        print("CUDA is available.")
        gpu_count = torch.cuda.device_count()
        device = torch.device("cuda")
        print(f"Using GPU device(s): {device}. Total GPUs: {gpu_count}")
    else:
        print("CUDA is not available.")
        device = torch.device("cpu")

    print(f"Predicting on dataset {dataset_dir}. Loading model weights from {weights}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # Normalize to mean and standard deviations of RGB values for ImageNet
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = DolphinOrientationDataset(dataset_root_dir=dataset_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=num_workers)

    model = OrientationResNet()
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    model.to(device)

    model.eval()
    all_outputs = []
    all_indices = []
    all_tracks = []

    print(f"Model loaded. Predicting on {len(dataset)} images.")
    with torch.no_grad():
        for images, _, tracks, idxs in tqdm(dataloader, desc="Predicting", unit="batch"):
            images = images.to(device)
            outputs = model(images)
            all_outputs.append(outputs)
            all_indices.append(idxs)
            all_tracks.append(tracks)

    all_outputs = torch.cat(all_outputs, dim=0).cpu()
    all_indices = torch.cat(all_indices, dim=0).cpu().numpy()
    all_tracks = torch.cat(all_tracks, dim=0).cpu().numpy()
    all_filenames = [str(dataset.get_image_path(idx).stem) for idx in all_indices]

    print(f"Predictions complete. Saving to {outfile_path}")

    # Create a DataFrame
    data = {
        'dataloader_index': all_indices,
        'filename': all_filenames,
        'object_id': all_tracks,
    }
    other_df = pd.DataFrame(data)

    model.write_outputs(all_outputs, other_df, outfile_path)
    print(f"Final angles saved to {outfile_path}")

if __name__ == "__main__":
    main()
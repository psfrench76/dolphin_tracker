import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.inc.orientation_network import OrientationResNet
from utils.inc.orientation_dataset import DolphinOrientationDataset
from utils.inc.settings import set_seed, settings
from tqdm import tqdm
import pandas as pd

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, targets, _, _ in dataloader:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss

def main():
    parser = argparse.ArgumentParser(description="Train the orientation model.")
    parser.add_argument('--dataset', type=Path, required=True, help="Path to the dataset root directory.")
    parser.add_argument('--output_folder', type=Path, required=True, help="Path to the output folder.")
    args = parser.parse_args()

    dataset_root_dir = args.dataset
    output_folder = args.output_folder
    outfile_path = output_folder / f"{output_folder.name}_{settings['orientations_results_suffix']}"
    weights_file_path = output_folder / settings['orientations_weights_file']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(0)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = DolphinOrientationDataset(dataset_root_dir=dataset_root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    model = OrientationResNet().to(device)
    criterion = model.compute_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 150
    for epoch in range(num_epochs):
        epoch_loss = train(model, dataloader, criterion, optimizer, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    torch.save(model.state_dict(), weights_file_path)
    print(f"Model saved to {weights_file_path}")

    # Save the final angles to a file
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
    data = {'dataloader_index': all_indices, 'filename': all_filenames, 'object_id': all_tracks}
    other_df = pd.DataFrame(data)

    model.write_outputs(all_outputs, other_df, outfile_path)
    print(f"Final angles saved to {outfile_path}")

if __name__ == "__main__":
    main()
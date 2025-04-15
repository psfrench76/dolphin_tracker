import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from utils.inc.orientation_network import OrientationResNet
from utils.inc.orientation_dataset import DolphinOrientationDataset
from utils.inc.settings import set_seed, settings, project_path, get_device_and_workers
import pandas as pd
import yaml
import time

def main():
    parser = argparse.ArgumentParser(description="Train the orientation model.")
    parser.add_argument('--data_name', '-d', type=str, required=True, help="Name of the dataset configuration file.")
    parser.add_argument('--output_folder', '-o', type=Path, required=True, help="Path to the output folder.")
    parser.add_argument('--hyp_path', '-h', type=str, default='default', help="Path to the hyperparameters file.")
    args = parser.parse_args()

    data_config_path = project_path(f"cfg/data/{args.data_name}.yaml")
    # TODO: make this more robust
    output_folder = args.output_folder
    if args.hyp_path == 'default':
        hyp_path = None
    else:
        hyp_path = Path(args.hyp_path)
        # TODO: Hyperparameter config parsing
        raise NotImplementedError("Hyperparameter config parsing not implemented yet.")

    outfile_path = output_folder / f"{output_folder.name}_{settings['orientations_results_suffix']}"
    weights_file_path = output_folder / settings['orientations_weights_file']

    output_folder.mkdir(parents=True, exist_ok=True)

    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir=output_folder)

    with open(data_config_path, 'r') as file:
        data_config = yaml.safe_load(file)

    dataset_root_dir = Path(data_config['path'])

    if not dataset_root_dir.is_dir():
        dataset_root_dir = Path(str(dataset_root_dir).lstrip('../')) # Ultralytics configs have an extra ../
        if not dataset_root_dir.is_dir():
            raise FileNotFoundError(f"Dataset root directory {dataset_root_dir} does not exist.")

    device, num_workers = get_device_and_workers()
    set_seed(0)

    # Ultralytics likes the dataset to point to the "images" folder, but I don't like that. In the interest of
    # maintaining the config format, I will .parent them.
    train_data_path = (dataset_root_dir / data_config['train']).parent
    val_data_path = (dataset_root_dir / data_config['val']).parent

    train_dataset = DolphinOrientationDataset(dataset_root_dir=train_data_path)
    val_dataset = DolphinOrientationDataset(dataset_root_dir=val_data_path)

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=num_workers)

    model = OrientationResNet()
    model.set_device(device)

    lr_start = 0.001
    lr_final_factor = 0.01
    num_epochs = 100
    lr_end = lr_start * lr_final_factor

    optimizer = torch.optim.Adam(model.parameters(), lr=lr_start)


    # Calculate the lambda function for the learning rate schedule
    def lr_lambda(epoch):
        if num_epochs == 1:
            return lr_start
        else:
            return lr_end / lr_start + (1 - lr_end / lr_start) * (1 - epoch / (num_epochs - 1))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)


    for e in range(1, num_epochs+1):
        start_time = time.time()  # Start time
        epoch_loss = model.train_model(train_dataloader, optimizer)
        val_loss = model.validate_model(val_dataloader)

        # Log the losses to TensorBoard
        writer.add_scalar('Loss/Train', epoch_loss, e)
        writer.add_scalar('Loss/Validation', val_loss, e)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], e)

        scheduler.step()

        end_time = time.time()  # End time
        epoch_duration = end_time - start_time
        print(f"Epoch {e}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}. Duration: {epoch_duration:.2f} seconds")

    torch.save(model.state_dict(), weights_file_path)
    print(f"Model saved to {weights_file_path}")

    all_outputs, all_indices, all_tracks = model.predict(val_dataloader)
    all_filenames = [str(val_dataset.get_image_path(idx).stem) for idx in all_indices]

    print(f"Predictions complete. Saving to {outfile_path}")

    data = {'dataloader_index': all_indices, 'filename': all_filenames, 'object_id': all_tracks}
    other_df = pd.DataFrame(data)

    model.write_outputs(all_outputs, other_df, outfile_path)
    print(f"Final angles saved to {outfile_path}")

    # Close the TensorBoard writer
    writer.close()

if __name__ == "__main__":
    main()
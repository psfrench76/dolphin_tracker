import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR
from utils.inc.orientation_network import OrientationResNet
from utils.inc.orientation_dataset import DolphinOrientationDataset
from utils.inc.settings import set_seed, settings, project_path, get_device_and_workers, storage_path
from utils.inc.reporting import OrientationMetrics
import pandas as pd
import yaml
import time


def main():
    parser = argparse.ArgumentParser(description="Train the orientation model.")
    parser.add_argument('--data_name', '-d', type=str, required=True, help="Name of the dataset configuration file.")
    parser.add_argument('--output_folder', '-o', type=Path, required=True, help="Path to the output folder.")
    parser.add_argument('--hyp_path', '-hyp', type=Path, default='default', help="Path to the hyperparameters file within the cfg/hyp/orientations directory")
    parser.add_argument('--augment', '-a', action='store_true', help="Use data augmentation.")
    args = parser.parse_args()

    data_config_path = project_path(f"cfg/data/{args.data_name}.yaml")
    output_folder = args.output_folder
    if args.hyp_path == 'default':
        hyp_path = project_path("cfg/hyp/orientations") / "default.yaml"
    else:
        hyp_path = project_path("cfg/hyp/orientations") / args.hyp_path

    if hyp_path.is_file():
        with open(hyp_path, 'r') as file:
            hp = yaml.safe_load(file)
    else:
        raise FileNotFoundError(f"Hyperparameter configuration file {hyp_path} does not exist.")

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

    device, num_workers = get_device_and_workers(split=False)
    set_seed(0)

    # Ultralytics likes the dataset to point to the "images" folder, but I don't like that. In the interest of
    # maintaining the config format, I will .parent them.
    train_data_path = (dataset_root_dir / data_config['train']).parent
    val_data_path = (dataset_root_dir / data_config['val']).parent

    imgsz = int(hp['imgsz'])

    augment = args.augment or bool(hp['augment'])

    train_dataset = DolphinOrientationDataset(dataset_root_dir=train_data_path, augment=augment, imgsz=imgsz)
    val_dataset = DolphinOrientationDataset(dataset_root_dir=val_data_path, augment=augment, imgsz=imgsz)

    dataloader_args = {
        'batch_size': int(hp['batch_size']),
        'num_workers': num_workers,
        'pin_memory': True,
        'shuffle': True,
        'prefetch_factor': 4,
    }

    train_dataloader = DataLoader(train_dataset, **dataloader_args)
    val_dataloader = DataLoader(val_dataset, **dataloader_args)
    print(f"Initializing model with {len(train_dataset)} training images and {len(val_dataset)} validation images.")

    if 'loss' in hp:
        loss = hp['loss']
    else:
        loss = 'MSE'

    model = OrientationResNet(loss=loss)
    model.set_device(device)

    lr_start = float(hp['lr_start'])
    lr_final_factor = float(hp['lr_final_factor'])
    num_epochs = int(hp['num_epochs'])
    freeze = int(hp['freeze'])
    lr_end = lr_start * lr_final_factor


    model.freeze_layers(freeze)
    print(f"Initializing optimizer")
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_start)

    # Calculate the lambda function for the learning rate schedule
    def lr_lambda(epoch):
        if num_epochs == 1:
            return lr_start
        else:
            return lr_end / lr_start + (1 - lr_end / lr_start) * (1 - epoch / (num_epochs - 1))

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    print(f"Training for {num_epochs} epochs")
    for e in range(1, num_epochs+1):
        start_time = time.time()  # Start time

        epoch_loss = model.train_model_scaled(train_dataloader, optimizer)
        val_loss = model.validate_model(val_dataloader)

        # Log the losses to TensorBoard
        writer.add_scalar('Loss/Train', epoch_loss, e)
        writer.add_scalar('Loss/Validation', val_loss, e)
        writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], e)

        scheduler.step()

        end_time = time.time()  # End time
        epoch_duration = end_time - start_time
        print(f"Epoch {e}/{num_epochs}, Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}. Duration: {epoch_duration:.2f} seconds. Train cache size: {train_dataset.get_cache_size_mb():.2f} MB")

    torch.save(model.state_dict(), weights_file_path)
    print(f"Model saved to {weights_file_path}")

    om = model.evaluate(val_dataloader, outfile_path)

    metrics_file = output_folder / f"{output_folder.name}_{settings['orientations_metrics_suffix']}"

    om.write_results(metrics_file)
    om.print_results()

    print(f"Metrics saved to {metrics_file}")

    # Close the TensorBoard writer
    writer.close()

if __name__ == "__main__":
    main()
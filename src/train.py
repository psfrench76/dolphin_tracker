from ultralytics import YOLO
import click
import yaml
import datetime
import os
import torch
import glob
import shutil
import multiprocessing
import psutil
import timeit
from torch.utils.tensorboard import SummaryWriter

@click.command()
@click.option('--model_name', default=None, help='Path to the model configuration file.')
@click.option('--data_name', default=None, help='Path to the data configuration file.')
def train_detector(model_name, data_name):
    config = 'cfg/settings.yaml'
    phase = 'train'
    print(f"Loading configuration files...")

    with open(config, 'r') as file:
        settings = yaml.safe_load(file)

    if model_name is None:
        model_config = settings['model_config_default']
    else:
        model_config = os.path.join(settings['model_config_dir'], model_name) + '.yaml'

    if data_name is None:
        data_config = settings['data_config_default']
    else:
        data_config = os.path.join(settings['data_config_dir'], data_name) + '.yaml'

    with open(model_config, 'r') as file:
        model_settings = yaml.safe_load(file)
        print(f"Found model configuration file {model_config} and successfully loaded")

    with open(data_config, 'r') as file:
        data_settings = yaml.safe_load(file)
        print(f"Found data configuration file {data_config} and successfully loaded")

    pretrained_weights_file = model_settings['train']['pretrained_model']
    epochs = settings['epochs']
    project = settings[phase]['project']

    run_number = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    name = f'{model_name}/{data_name}/{run_number}'

    # Detect the number of available CPU cores
    num_cores = len(psutil.Process().cpu_affinity())

    # Set the number of workers to the recommended value
    num_workers = int(min(16, num_cores) / 2)

    print(f"Number of available CPU cores: {num_cores}")
    print(f"Setting number of workers to: {num_workers} (divided by 2 for train/val split)")

    # Check for CUDA availability
    if torch.cuda.is_available():
        print("CUDA is available.")
    else:
        print("CUDA is not available.")

    print(f"Loading pretrained weights from {pretrained_weights_file}")
    model = YOLO(pretrained_weights_file)

    now = timeit.default_timer()
    print(f"Training model {model_name} on data {data_name} for {epochs} epochs")
    results = model.train(data=data_config, epochs=epochs, project=project, name=name, workers=num_workers)
    print(f"Training completed in {timeit.default_timer() - now} seconds using {num_workers} workers")

    # Check for .pt files and move them
    amp_check_models_dir = settings['amp_check_models_dir']
    for pt_file in glob.glob("*.pt"):
        if not os.path.islink(pt_file):
            new_path = os.path.join(amp_check_models_dir, pt_file)
            shutil.move(pt_file, new_path)
            os.symlink(new_path, pt_file)

if __name__ == '__main__':
    train_detector()
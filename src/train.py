from ultralytics import YOLO
import click
import yaml
import datetime
import os
import torch
import glob
import shutil
import psutil
import timeit

@click.command()
@click.option('--model_name', default=None, help='Path to the model configuration file.')
@click.option('--data_name', default=None, help='Path to the data configuration file.')
def train_detector(model_name, data_name):
    print(f"Loading configuration files...")
    config = 'cfg/settings.yaml'
    phase = 'train'

    with open(config, 'r') as file:
        settings = yaml.safe_load(file)

    model_name = model_name or settings['model_config_default']
    model_config = os.path.join(settings['model_config_dir'], model_name) + '.yaml'
    with open(model_config, 'r') as file:
        model_settings = yaml.safe_load(file)
        print(f"Found model configuration file {model_config} and successfully loaded")

    data_name = data_name or settings['data_config_default']
    data_config = os.path.join(settings['data_config_dir'], data_name) + '.yaml'
    with open(data_config, 'r') as file:
        data_settings = yaml.safe_load(file)
        print(f"Found data configuration file {data_config} and successfully loaded")

    pretrained_file = model_settings['pretrained_model']
    project = os.path.join(settings['runs_dir'], phase)

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
        gpu_count = torch.cuda.device_count()
        device = [i for i in range (gpu_count)]
        print("Using GPU device(s):", device)
    else:
        print("CUDA is not available.")
        device = 'cpu'

    os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

    print(f"Loading pretrained weights from {pretrained_file}")
    model = YOLO(pretrained_file)

    now = timeit.default_timer()
    print(f"Training model {model_name} on data {data_name}")
    results = model.train(data=data_config, project=project, name=name, workers=num_workers,
                            cfg=model_settings['hyp'], device=device)
    print(f"Training completed in {timeit.default_timer() - now} seconds using {num_workers} workers")

    # Check for downloaded .pt files and move them out of the current directory
    amp_check_models_dir = settings['amp_check_models_dir']
    for pt_file in glob.glob("*.pt"):
        if not os.path.islink(pt_file):
            new_path = os.path.join(amp_check_models_dir, pt_file)
            shutil.move(pt_file, new_path)
            os.symlink(new_path, pt_file)

if __name__ == '__main__':
    train_detector()
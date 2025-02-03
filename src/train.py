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
@click.option('--data_name', required=True, help='Name of the data configuration file.')
@click.option('--run_name', required=True, help='Name of the run. Expected: exp##/param-desc')
@click.option('--hyp_path', required=True, help='Path to the hyperparameter configuration file.')
@click.option('--weights_path', required=True, help='Path to the pretrained weights file.')
def train_detector(data_name, run_name, hyp_path, weights_path):
    print(f"Loading configuration files...")
    config = 'cfg/settings.yaml'
    phase = 'train'

    with open(config, 'r') as file:
        settings = yaml.safe_load(file)

    data_config = os.path.join(settings['data_config_dir'], data_name) + '.yaml'

    project = os.path.join(settings['runs_dir'], phase, 'stage1')

    run_number = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    name = f'{run_name}/{run_number}'

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

    print(f"Loading pretrained weights from {weights_path}")
    model = YOLO(weights_path)

    now = timeit.default_timer()
    print(f"Training model {run_name} on data {data_name}")
    results = model.train(data=data_config, project=project, name=name, workers=num_workers,
                            cfg=hyp_path, device=device)
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
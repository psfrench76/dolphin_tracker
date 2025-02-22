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
import time

@click.command()
@click.option('--data_name', required=True, help='Name of the data configuration file.')
@click.option('--run_name', required=True, help='Name of the run. Expected: exp##/param-desc')
@click.option('--hyp_path', required=True, help='Path to the hyperparameter configuration file.')
@click.option('--weights_path', required=True, help='Path to the pretrained weights file.')
@click.option('--checkpoint_reload', is_flag=True, help='If set, save every epoch and reload from the last checkpoint.')
def train_detector(data_name, run_name, hyp_path, weights_path, checkpoint_reload):
    print(f"Loading configuration files...")
    config = 'cfg/settings.yaml'
    phase = 'train'

    with open(config, 'r') as file:
        settings = yaml.safe_load(file)

    project = os.path.join(settings['runs_dir'], phase, 'stage1')
    run_number = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    resume = False

    #save_period = 1 if checkpoint_reload else -1
    save_period = -1 # I think save_period is unnecessary for preemption, but here JIC. I think last.pt always saved.

    if checkpoint_reload:
        print("Checking for checkpoint file...")
        run_dir = os.path.join(project, run_name)
        checkpoint_file = os.path.join(run_dir, settings['checkpoint_file'])
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file, 'r') as file:
                checkpoint_settings = yaml.safe_load(file)
                run_number = checkpoint_settings['run_number']
                if run_number is None:
                    raise ValueError(f"Checkpoint file {checkpoint_file} exists but does not contain a run number.")

                last_path = os.path.join(project, run_name, run_number, 'weights', 'last.pt')
                if os.path.exists(last_path):
                    weights_path = last_path
                    resume = True
                else:
                    print(f"Checkpoint file indicated {last_path}, but weights file not found. Starting training with {weights_path} instead.")
                print(f"Loaded checkpoint file, resuming training under run number {run_number}")
        else:
            os.makedirs(run_dir, exist_ok=True)
            with open(checkpoint_file, 'w') as file:
                file.write(f"run_number: '{run_number}'\n")
                print(f"Checkpoint file not found, starting fresh run with run number {run_number}")

    name = f'{run_name}/{run_number}'

    with open(hyp_path, 'r') as file:
        hyp = yaml.safe_load(file)

    data_config = os.path.join(settings['data_config_dir'], data_name) + '.yaml'

    with open(data_config, 'r') as file:
        data_settings = yaml.safe_load(file)

    data_path = data_settings['path'].replace('../', '', 1)

    label_folders = [os.path.join(data_path, data_settings[split].replace('images', 'labels')) for split in
                     ['train', 'val']]
    new_label_folders = [os.path.join(data_path, data_settings[split].replace('images', 'original_labels'))
                         for split in ['train', 'val']]

    print("Processing labels...")

    for label_folder, new_label_folder in zip(label_folders, new_label_folders):
        process_labels(label_folder, new_label_folder)

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

    if resume:
        #Clear cache because it doesn't like reloading it here
        model.data = None

    now = timeit.default_timer()
    print(f"Training model {run_name} on data {data_name}")
    for attempts in range(1, 4):
        try:
            results = model.train(data=data_config, project=project, name=name, workers=num_workers,
                                  cfg=hyp_path, device=device, save_period=save_period, resume=resume,
                                  exist_ok=True)
            break
        except FileNotFoundError as e:
            if attempts >= 3:
                raise FileNotFoundError(f"Error '{e}' training model {run_name} on data {data_name} after 3 attempts.")
            else:
                print(f"Error training model {run_name} on data {data_name}, retrying in 5 seconds...")
                time.sleep(5)

    print(f"Training completed in {timeit.default_timer() - now} seconds using {num_workers} workers")

    # Check for downloaded .pt files and move them out of the current directory
    amp_check_models_dir = settings['amp_check_models_dir']
    for pt_file in glob.glob("*.pt"):
        if not os.path.islink(pt_file):
            new_path = os.path.join(amp_check_models_dir, pt_file)
            shutil.move(pt_file, new_path)
            os.symlink(new_path, pt_file)

def process_labels(label_folder, new_label_folder):
    """
    Move label files with class IDs other than 0 to a new folder and save a copy with class ID 0 in the original folder.
    """
    if not os.path.exists(new_label_folder):
        os.makedirs(new_label_folder)

    for root, _, files in os.walk(label_folder):
        for file in files:
            if file.endswith('.txt'):
                label_file_path = str(os.path.join(root, file))
                new_label_file_path = str(os.path.join(new_label_folder, file))
                for i in range(1,4):
                    try:
                        with open(label_file_path, 'r') as f:
                            lines = f.readlines()
                        break
                    except FileNotFoundError:
                        if i >= 3:
                            raise FileNotFoundError(f"Error reading {label_file_path} after 3 attempts.")
                        else:
                            print(f"Error reading {label_file_path}, retrying in 5 seconds...")
                            time.sleep(5)

                move_file = True
                new_lines = []
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) > 0:
                        if parts[0] != '0':
                            parts[0] = '0'

                        if float(parts[1]) < 0:
                            x_start = float(parts[1])
                            x_width = float(parts[3])
                            x_width += x_start
                            parts[3] = str(x_width)
                            parts[1] = '0'

                        if float(parts[2]) < 0:
                            y_start = float(parts[2])
                            y_height = float(parts[4])
                            y_height += y_start
                            parts[4] = str(y_height)
                            parts[2] = '0'

                    new_line = ' '.join(parts) + '\n'
                    if len(new_lines) == 0 or new_line != new_lines[-1]:
                        new_lines.append(new_line)

                if move_file:
                    shutil.move(label_file_path, new_label_file_path)
                    with open(label_file_path, 'w') as f:
                        f.writelines(new_lines)

if __name__ == '__main__':
    train_detector()
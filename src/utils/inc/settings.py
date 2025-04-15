"""
This module is used to load the settings from the settings.yaml file. It also loads user_settings from the file
specified by the user_settings_dir and user_settings_file settings. The user_settings file is used to store user
specific settings that should not be checked into version control. These settings will override the settings in the
settings.yaml file. The project_root variable is used to store the root directory of the project. The config_file

The settings are stored in a dictionary called settings which can be imported in other files.

The project_path function is used to create a path relative to the project root directory.
The storage_path function is used to create a path relative to the storage root directory.
"""

import yaml
from pathlib import Path

project_root = Path(__file__).resolve().parents[3]
config_file = project_root / 'cfg/settings.yaml'

with open(config_file, 'r') as f:
    settings = yaml.safe_load(f)

user_settings = {}
user_settings_path = project_root / settings['user_settings_dir'] / settings['user_settings_file']
if user_settings_path.is_file():
    with open(user_settings_path, 'r') as f:
        user_settings = yaml.safe_load(f)

if user_settings is not None:
    settings.update(user_settings)

storage_root = Path(settings['storage_root']).resolve()


def project_path(path):
    return project_root / path


def storage_path(path):
    return storage_root / path


def set_seed(seed):
    """
    Set the random seed for reproducibility.
    """
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device_and_workers(split=True):
    import psutil
    from torch import cuda, device
    # TODO - this (cuda and workers) is (almost) the same as in train.py. Let's modularize this stuff.
    #   Note that the UL module uses an array of numbers though. When modularizing make sure that mutli-GPU isn't broken for UL.
    num_cores = len(psutil.Process().cpu_affinity())

    # Set the number of workers to the recommended value
    if split:
        num_workers = int(min(16, num_cores) / 2)
    else:
        num_workers = int(min(16, num_cores))

    print(f"Number of available CPU cores: {num_cores}")
    print(f"Setting number of workers to: {num_workers}")

    # Check for CUDA availability
    if cuda.is_available():
        print("CUDA is available.")
        gpu_count = cuda.device_count()
        device = device("cuda")
        print(f"Using GPU device(s): {device}. Total GPUs: {gpu_count}")
    else:
        print("CUDA is not available.")
        device = device("cpu")

    return device, num_workers
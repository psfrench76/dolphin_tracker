import yaml
import os
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
config_file = project_root / 'cfg/settings.yaml'


def project_path(path):
    return project_root / path


with open(config_file, 'r') as f:
    settings = yaml.safe_load(f)

user_settings = {}
if os.path.exists('usr/user_settings.yaml'):
    with open('usr/user_settings.yaml', 'r') as f:
        user_settings = yaml.safe_load(f)

settings.update(user_settings)

storage_root = Path(settings['storage_root']).resolve()


def storage_path(path):
    return storage_root / path

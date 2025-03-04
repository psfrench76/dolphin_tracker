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

settings.update(user_settings)

storage_root = Path(settings['storage_root']).resolve()


def project_path(path):
    return project_root / path


def storage_path(path):
    return storage_root / path

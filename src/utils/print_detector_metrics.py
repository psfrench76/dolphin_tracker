"""
This script prints detector metrics from a results CSV file based on a specified SLURM job ID, run name,
and experiment number. It checks for the existence of necessary files and ensures that training is completed before
extracting and printing the metrics.

Usage: print_detector_metrics.py <slurm_job_id> <run_name> <experiment>
"""

import argparse
import pandas as pd
import yaml
import sys
from inc.settings import settings, storage_path


# Find the results CSV file based on the SLURM job ID, run name, and experiment number, and print the detector metrics.
def _find_and_print_results(slurm_job_id, run_name, experiment):
    runs_dir_path = storage_path(settings['detector_runs_dir']) / settings['training_runs_dir']
    this_run_dir_path = runs_dir_path / experiment / run_name
    checkpoint_file_path = this_run_dir_path / settings['checkpoint_file']

    logs_dir_path = storage_path(settings['logs_dir']) / settings['log_results_dir']
    log_file_path = logs_dir_path / f"{settings['training_log_prefix']}_{slurm_job_id}.log"

    # Check if checkpoint file exists
    if not checkpoint_file_path.exists():
        print(f"Error: The checkpoint file '{checkpoint_file_path}' does not exist.")
        sys.exit(1)

    # Check if log file exists
    if not log_file_path.exists():
        print(f"Error: The log file '{log_file_path}' does not exist.")
        sys.exit(1)

    # Check if training is completed
    with log_file_path.open('r') as log_file:
        log_lines = log_file.readlines()
        if not log_lines or not log_lines[-1].strip().startswith("Training completed in"):
            print("Error: Training is not completed.")
            sys.exit(1)

    # Load the checkpoint file to get the run number
    with checkpoint_file_path.open('r') as checkpoint_file:
        checkpoint_data = yaml.safe_load(checkpoint_file)
        run_number = checkpoint_data.get('run_number')
        if not run_number:
            print(f"Error: 'run_number' not found in the checkpoint file '{checkpoint_file_path}'.")
            sys.exit(1)

    results_csv_path = this_run_dir_path / run_number / settings['training_results_csv']
    if not results_csv_path.exists():
        print(f"Error: The file '{results_csv_path}' does not exist.")
        sys.exit(1)

    data = pd.read_csv(results_csv_path)

    # Check that all required metrics are present in the CSV file
    for metric in settings['training_metrics_to_print']:
        if metric not in data.columns:
            print(f"Error: The metric '{metric}' is not found in the file '{results_csv_path}'.")
            sys.exit(1)

    # Calculate the fitness function for each epoch
    data['fitness'] = 0.1 * data['metrics/mAP50(B)'] + 0.9 * data['metrics/mAP50-95(B)']

    # Get the epoch with the highest fitness value
    best_row = data.loc[data['fitness'].idxmax(), settings['training_metrics_to_print']]

    headers = settings['training_metrics_to_print'].copy()
    headers = ['Run number'] + headers

    values = list(map(str, best_row.values))
    values = [run_number] + values

    print(f"Printing metrics from the results CSV file '{results_csv_path}':")

    print("\t".join(headers))
    print("\t".join(values))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Print detector metrics from a results CSV file.")
    parser.add_argument('slurm_job_id', type=str, help='SLURM job ID')
    parser.add_argument('run_name', type=str, help='Run name')
    parser.add_argument('experiment', type=str, help='Experiment name')
    args = parser.parse_args()

    _find_and_print_results(args.slurm_job_id, args.run_name, args.experiment)

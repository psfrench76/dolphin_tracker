import os
import sys
import pandas as pd
import yaml

def find_results_csv(parent_dir, slurm_job_id, run_name, experiment):
    # Construct paths
    checkpoint_path = os.path.join(parent_dir, f"runs/train/stage1/{experiment}", run_name, "checkpoint.yaml")
    log_path = os.path.join(parent_dir, f"logs/results/train_{slurm_job_id}.log")

    # Check if checkpoint file exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: The checkpoint file '{checkpoint_path}' does not exist.")
        sys.exit(1)

    # Check if log file exists
    if not os.path.exists(log_path):
        print(f"Error: The log file '{log_path}' does not exist.")
        sys.exit(1)

    # Check if training is completed
    with open(log_path, 'r') as log_file:
        log_lines = log_file.readlines()
        if not log_lines or not log_lines[-1].strip().startswith("Training completed in"):
            print("Error: Training is not completed.")
            sys.exit(1)

    # Load the checkpoint file to get the run number
    with open(checkpoint_path, 'r') as checkpoint_file:
        checkpoint_data = yaml.safe_load(checkpoint_file)
        run_number = checkpoint_data.get('run_number')
        if not run_number:
            print(f"Error: 'run_number' not found in the checkpoint file '{checkpoint_path}'.")
            sys.exit(1)

    # Construct the path to the results.csv file
    results_csv_path = os.path.join(parent_dir, f"runs/train/stage1/{experiment}", run_name, run_number, "results.csv")

    # Check if results.csv file exists
    if not os.path.exists(results_csv_path):
        print(f"Error: The file '{results_csv_path}' does not exist.")
        sys.exit(1)

    return results_csv_path

def print_metrics(results_csv_path):
    # Read the CSV file
    data = pd.read_csv(results_csv_path)

    # Extract the required metrics
    metrics = [
        'metrics/precision(B)',
        'metrics/recall(B)',
        'metrics/mAP50(B)',
        'metrics/mAP50-95(B)',
        'val/box_loss',
        'val/cls_loss',
        'val/dfl_loss'
    ]

    # Check if all required metrics are present in the CSV file
    for metric in metrics:
        if metric not in data.columns:
            print(f"Error: The metric '{metric}' is not found in the file '{results_csv_path}'.")
            sys.exit(1)

    # Calculate the fitness function for each row
    data['fitness'] = 0.1 * data['metrics/mAP50(B)'] + 0.9 * data['metrics/mAP50-95(B)']

    # Get the row with the highest fitness value
    best_row = data.loc[data['fitness'].idxmax(), metrics]

    # Print the metrics in the specified order
    print("\t".join(metrics))
    print("\t".join(map(str, best_row.values)))

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python detector_metrics.py <slurm_job_id> <run_name> <experiment>")
        sys.exit(1)

    slurm_job_id = sys.argv[1]
    run_name = sys.argv[2]
    experiment = sys.argv[3]

    parent_dir = os.path.dirname(os.getcwd())
    results_csv_path = find_results_csv(parent_dir, slurm_job_id, run_name, experiment)
    print_metrics(results_csv_path)
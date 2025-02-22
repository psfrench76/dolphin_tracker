import os
import sys
import subprocess
import re
import yaml
import csv
from datetime import datetime, timedelta

def get_squ_results():
    result = subprocess.run(['squ'], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error: Failed to run 'squ' command.")
        sys.exit(1)
    squ_results = {}
    for line in result.stdout.splitlines():
        match = re.search(r'^\s*(\d+)\s+\S+\s+\S+\s+\S+\s+(\S+)', line)
        if match:
            job_id, job_status = match.groups()
            squ_results[job_id] = job_status
    return squ_results

def get_slurm_job_id(run_name, squ_results):
    for job_id, job_status in squ_results.items():
        log_path = os.path.expanduser(f"~/logs/dolphin-tracker/results/train_{job_id}.log")
        if os.path.exists(log_path):
            with open(log_path, 'r') as file:
                for log_line in file:
                    if f"save_dir=../runs/train/stage1/{run_name}" in log_line:
                        return job_id, job_status

    log_dir = os.path.expanduser("~/logs/dolphin-tracker/slurm")
    four_days_ago = datetime.now() - timedelta(days=4)
    for log_file in os.listdir(log_dir):
        if log_file.startswith("train_") and log_file.endswith(".out"):
            log_path = os.path.join(log_dir, log_file)
            if datetime.fromtimestamp(os.path.getmtime(log_path)) >= four_days_ago:
                with open(log_path, 'r') as file:
                    for line in file:
                        if f"run_name = {run_name}" in line:
                            job_id = re.search(r'train_(\d+)\.out', log_file).group(1)
                            return job_id, squ_results.get(job_id, None)

    return None, None

def check_status(run_name):
    squ_results = get_squ_results()
    job_id, job_status = get_slurm_job_id(run_name, squ_results)
    if job_id is None:
        run_dir = os.path.join("..", "runs", "train", "stage1", run_name)
        checkpoint_path = os.path.join(run_dir, "checkpoint.yaml")
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'r') as file:
                checkpoint_data = yaml.safe_load(file)
                run_number = checkpoint_data.get('run_number')
                if run_number:
                    run_folder = os.path.join(run_dir, run_number)
                    results_csv_path = os.path.join(run_folder, "results.csv")
                    args_yaml_path = os.path.join(run_folder, "args.yaml")
                    if os.path.exists(results_csv_path) and os.path.exists(args_yaml_path):
                        with open(args_yaml_path, 'r') as file:
                            args_data = yaml.safe_load(file)
                            num_epochs = args_data.get('epochs')
                            if num_epochs:
                                with open(results_csv_path, 'r') as file:
                                    last_line = list(csv.reader(file))[-1]
                                    if last_line and last_line[0].isdigit() and int(last_line[0]) == num_epochs:
                                        print("Status: Completed")
                                        return

        print("Status: Job Failed, Missing, or Waiting to Start")
        return

    log_path = os.path.expanduser(f"~/logs/dolphin-tracker/results/train_{job_id}.log")
    err_path = os.path.expanduser(f"~/logs/dolphin-tracker/slurm/train_{job_id}.err")

    if not os.path.exists(log_path):
        print(f"Error: Log file '{log_path}' does not exist.")
        sys.exit(1)

    if not os.path.exists(err_path):
        print(f"Error: Error file '{err_path}' does not exist.")
        sys.exit(1)

    with open(log_path, 'r') as log_file:
        log_lines = log_file.readlines()
        if log_lines and log_lines[-1].strip().startswith("Training completed in"):
            print("Status: Completed")
            return

    if job_status == 'R':
        log_mtime = datetime.fromtimestamp(os.path.getmtime(log_path))
        if datetime.now() - log_mtime <= timedelta(minutes=30):
            run_dir = os.path.join("..", "runs", "train", "stage1", run_name)
            checkpoint_path = os.path.join(run_dir, "checkpoint.yaml")
            if os.path.exists(checkpoint_path):
                with open(checkpoint_path, 'r') as file:
                    checkpoint_data = yaml.safe_load(file)
                    run_number = checkpoint_data.get('run_number')
                    if run_number:
                        run_folder = os.path.join(run_dir, run_number)
                        results_csv_path = os.path.join(run_folder, "results.csv")
                        if os.path.exists(results_csv_path):
                            with open(results_csv_path, 'r') as file:
                                last_line = list(csv.reader(file))[-1]
                                if last_line and last_line[0].isdigit():
                                    last_epoch_time = datetime.fromtimestamp(os.path.getmtime(results_csv_path))
                                    print(f"Status: Running and healthy, Most recent epoch: {last_line[0]}, Last epoch time: {last_epoch_time}")
                                    return
            print("Status: Running and healthy")
            return

    if job_status == 'R':
        job_start_time = datetime.fromtimestamp(os.path.getctime(log_path))
        if datetime.now() - job_start_time <= timedelta(minutes=30):
            print("Status: Restarting")
            return

    if job_status == 'PD':
        with open(err_path, 'r') as err_file:
            err_lines = err_file.readlines()
            if err_lines and err_lines[-1].strip().endswith("DUE TO PREEMPTION ***"):
                print("Status: Preempted and waiting")
                return

    if job_status is None:
        with open(err_path, 'r') as err_file:
            err_lines = err_file.readlines()
            if err_lines and err_lines[-1].strip().startswith("FileNotFoundError"):
                print("Status: Failed due to missing label (race condition)")
                return

    print(f"Status: Unknown. Job ID: {job_id}, Status: {job_status}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python check_train_status.py <run_name>")
        sys.exit(1)

    run_name = sys.argv[1]
    check_status(run_name)
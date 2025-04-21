import os
import subprocess
import click


def validate_args(dataset, model, tracker):
    if not os.path.exists(dataset):
        raise ValueError(f"Dataset folder {dataset} does not exist.")
    if not os.path.exists(model):
        raise ValueError(f"Model weights file {model} does not exist.")
    if tracker is not None and not os.path.exists(tracker):
        raise ValueError(f"Tracker configuration file {tracker} does not exist.")


@click.command()
@click.option('--dataset', required=True, help="Path to the dataset directory.")
@click.option('--model', required=True, help="Path to the model file.")
@click.option('--output', required=True, help="Path to the output directory.")
@click.option('--tracker', help="Tracker config file")
@click.option('-pv', is_flag=True, help="Generate prediction video.")
def main(dataset, model, output, tracker, pv):
    validate_args(dataset, model, tracker)

    # Set environment variables
    os.environ['DATASET'] = dataset
    os.environ['MODEL'] = model
    os.environ['OUTPUT'] = output
    os.environ['TRACKER'] = tracker if tracker else "Default"
    os.environ['PV'] = "-pv" if pv else ""

    subprocess.run(["sbatch", "utils/hpc/tfv_job.sbatch"])
    print("Job submitted.")


if __name__ == "__main__":
    main()

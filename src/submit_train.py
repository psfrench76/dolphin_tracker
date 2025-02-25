import os
import subprocess
import click


def validate_args(data_name, run_name, hyp_path, weights_path):
    if not os.path.exists(hyp_path):
        raise ValueError(f"Hyperparameter configuration file {hyp_path} does not exist.")
    if not os.path.exists(weights_path):
        raise ValueError(f"Pretrained weights file {weights_path} does not exist.")


@click.command()
@click.option('--data_name', required=True, help='Name of the data configuration file.')
@click.option('--run_name', required=True, help='Name of the run. Expected: exp##/param-desc')
@click.option('--hyp_path', required=True, help='Path to the hyperparameter configuration file.')
@click.option('--weights_path', required=True, help='Path to the pretrained weights file.')
@click.option('--checkpoint_reload', is_flag=True, help='If set, save every epoch and reload from the last checkpoint.')
def main(data_name, run_name, hyp_path, weights_path, checkpoint_reload):
    validate_args(data_name, run_name, hyp_path, weights_path)

    # Set environment variables
    os.environ['DATA_NAME'] = data_name
    os.environ['RUN_NAME'] = run_name
    os.environ['HYP_PATH'] = hyp_path
    os.environ['WEIGHTS_PATH'] = weights_path

    if checkpoint_reload:
        subprocess.run(["sbatch", "utils/hpc/train_preempt.sbatch"])
        print("Job submitted to preempt partition.")
    else:
        subprocess.run(["sbatch", "utils/hpc/train_job.sbatch"])
        print("Job submitted.")


if __name__ == "__main__":
    main()

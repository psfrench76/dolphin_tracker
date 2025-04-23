import os
import subprocess
import click


def validate_args(data_name, output_folder, hyp_path, augment):
    if not os.path.exists(hyp_path):
        raise ValueError(f"Hyperparameter configuration file {hyp_path} does not exist.")


@click.command()
@click.option('--data_name', '-d', required=True, help='Name of the data configuration file.')
@click.option('--output_folder', '-o', required=True, help='Path to the output folder.')
@click.option('--hyp_path', '-hyp', default='default', help='Path to the hyperparameter configuration file.')
@click.option('--augment', '-a', is_flag=True, help='Use data augmentation.')
def main(data_name, output_folder, hyp_path, augment):
    # validate_args(data_name, output_folder, hyp_path, augment)

    # Set environment variables
    os.environ['DATA_NAME'] = data_name
    os.environ['OUTPUT_FOLDER'] = output_folder
    os.environ['HYP_PATH'] = hyp_path
    os.environ['AUGMENT'] = '--augment' if augment else ''

    subprocess.run(["sbatch", "utils/hpc/orient_train_job.sbatch"])
    print("Job submitted.")

if __name__ == "__main__":
    main()

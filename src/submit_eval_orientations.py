import os
import subprocess
import click


def validate_args(dataset, output_folder, weights, augment):
    if not os.path.exists(dataset):
        raise ValueError(f"Dataset folder {dataset} does not exist.")

    if not os.path.exists(weights):
        raise ValueError(f"Weights file {weights} does not exist.")


@click.command()
@click.option('--dataset', '-d', required=True, help='Path to the dataset root directory.')
@click.option('--output_folder', '-o', required=True, help='Path to the output folder.')
@click.option('--weights', '-w', default='default', help='PPath to the model weights file.')
@click.option('--augment', '-a', is_flag=True, help='Use data augmentation.')
def main(dataset, output_folder, weights, augment):
    validate_args(dataset, output_folder, weights, augment)

    # Set environment variables
    os.environ['DATASET'] = dataset
    os.environ['OUTPUT_FOLDER'] = output_folder
    os.environ['WEIGHTS'] = weights
    os.environ['AUGMENT'] = '--augment' if augment else ''

    subprocess.run(["sbatch", "utils/hpc/orient_eval_job.sbatch"])
    print("Job submitted.")

if __name__ == "__main__":
    main()

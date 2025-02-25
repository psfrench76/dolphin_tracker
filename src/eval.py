import click
import torch
import psutil
from ultralytics import YOLO
from pathlib import Path

if __package__ is None or __package__ == '':
    from utils.settings import settings, project_path, storage_path
else:
    from .utils.settings import settings, project_path, storage_path


@click.command()
@click.option('--model_path', required=True, help='Path to the trained model file.')
@click.option('--data_name', required=True, help='Name of the dataset configuration file.')
@click.option('--output_dir', required=True, help='Directory to save the evaluation results.')
@click.option('--split', default='val', help='Split of the dataset to evaluate on (train, val, test).')
def main(model_path, data_name, output_dir, split):
    evaluate_model(model_path, data_name, output_dir, split)

def evaluate_model(model_path, data_name, output_path, split):
    phase = 'eval'
    model_path = Path(model_path)
    output_path = Path(output_path)

    data_config_path = project_path(f"cfg/data/{data_name}.yaml")

    project_dir_path = str(storage_path(f"runs/{phase}"))
    run_subdirectory = str(output_path.resolve().relative_to(storage_path('output').resolve()))

    print(f"Loading model from {model_path}")
    model = YOLO(model_path)

    if torch.cuda.is_available():
        print("CUDA is available.")
        device = 'cuda'
    else:
        print("CUDA is not available.")
        device = 'cpu'

    model.to(device)

    # Detect the number of available CPU cores
    num_cores = len(psutil.Process().cpu_affinity())

    # Set the number of workers to the recommended value
    num_workers = min(16, num_cores)

    print(f"Number of available CPU cores: {num_cores}")
    print(f"Setting number of workers to: {num_workers}")

    print(f"Evaluating model on {split} split...")
    results = model.val(data=data_config_path, split=split, project=project_dir_path, name=run_subdirectory, workers=num_workers)

    output_path.mkdir(parents=True, exist_ok=True)
    results_path = output_path / settings['evaluation_results_file']

    with open(results_path, 'w') as file:
        file.write(f"Results: {results}\n")

    print(f"Evaluation results saved to {results_path}")

if __name__ == '__main__':
    main()
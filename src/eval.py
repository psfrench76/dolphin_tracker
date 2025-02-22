import os
import click
import yaml
import torch
import psutil
from ultralytics import YOLO

@click.command()
@click.option('--model_path', required=True, help='Path to the trained model file.')
@click.option('--data_name', required=True, help='Name of the dataset configuration file.')
@click.option('--output_dir', required=True, help='Directory to save the evaluation results.')
@click.option('--split', default='val', help='Split of the dataset to evaluate on (train, val, test).')
def evaluate_model(model_path, data_name, output_dir, split):
    print(f"Loading configuration files...")
    config = 'cfg/settings.yaml'
    phase = 'eval'

    with open(config, 'r') as file:
        settings = yaml.safe_load(file)

    data_config = os.path.join(settings['data_config_dir'], data_name) + '.yaml'

    project = os.path.join(settings['runs_dir'], phase)
    name = output_dir.replace('../output/', '')

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
    results = model.val(data=data_config, split=split, project=project, name=name, workers=num_workers)

    os.makedirs(output_dir, exist_ok=True)
    results_path = str(os.path.join(output_dir, 'evaluation_results.txt'))

    with open(results_path, 'w') as file:
        file.write(f"Results: {results}\n")

    print(f"Evaluation results saved to {results_path}")

if __name__ == '__main__':
    evaluate_model()
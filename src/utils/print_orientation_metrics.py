import os
import sys
import pandas as pd


def print_metrics(results_csv):
    if not os.path.exists(results_csv):
        print(f"Error: The file '{results_csv}' does not exist.")
        sys.exit(1)

    # Read the CSV file
    data = pd.read_csv(results_csv)

    # Extract the required metrics
    metrics = [
        'MAE_Cos',
        'MeanAngularDistance',
        'MeanVectorNormError'
    ]

    # Check if all required metrics are present in the CSV file
    for metric in metrics:
        if metric not in data.columns:
            print(f"Error: The metric '{metric}' is not found in the file '{results_csv}'.")
            sys.exit(1)

    # Get the only row of the required metrics
    row = data[metrics].iloc[0]

    # Print the metrics in the specified order
    print("\t".join(metrics))
    print("\t".join(map(str, row.values)))


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python src/utils/print_orientation_metrics.py <results_csv>")
        sys.exit(1)

    results_csv = sys.argv[1]
    print_metrics(results_csv)

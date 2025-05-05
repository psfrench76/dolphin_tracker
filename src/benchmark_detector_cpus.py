import argparse
import time
from pathlib import Path
from predict import predict

def benchmark_predict(args, num_workers):
    """Run the predict function with a specific num_workers count and measure execution time."""
    start_time = time.time()
    predict(
        model_path=args.model_path,
        data_path=args.data_path,
        output_path=args.output_dir,
        split=args.split,
        num_workers=num_workers
    )
    elapsed_time = time.time() - start_time
    return elapsed_time

def main():
    parser = argparse.ArgumentParser(description="Benchmark detector with varying num_workers counts.")

    # Arguments passed through to predict.py
    parser.add_argument("--model_path", type=Path, required=True, help="Path to the model file.")
    parser.add_argument("--data_path", type=Path, required=True, help="Path to the dataset directory.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Path to the output directory.")
    parser.add_argument("--split", type=str, help="Dataset split to predict on (e.g., 'train', 'val', 'test').")

    # Benchmark-specific arguments
    parser.add_argument("--min_workers", type=int, default=1, help="Minimum number of workers to benchmark.")
    parser.add_argument("--max_workers", type=int, default=8, help="Maximum number of workers to benchmark.")

    args = parser.parse_args()

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Benchmarking from {args.min_workers} to {args.max_workers} workers...")
    results = []

    for num_workers in range(args.min_workers, args.max_workers + 1):
        print(f"Running benchmark with num_workers={num_workers}...")
        elapsed_time = benchmark_predict(args, num_workers)
        results.append((num_workers, elapsed_time))
        print(f"num_workers={num_workers}, time={elapsed_time:.2f} seconds")

    # Save results to a file
    results_file = args.output_dir / "benchmark_results.txt"
    with open(results_file, "w") as f:
        for num_workers, elapsed_time in results:
            f.write(f"num_workers={num_workers}, time={elapsed_time:.2f} seconds\n")

    print(f"Benchmarking complete. Results saved to {results_file}")

if __name__ == "__main__":
    main()
import argparse
import pandas as pd
import os

datasets = ['MSL', 'SMAP', 'SMD']

def chooseBestDetector(data: pd.DataFrame, metric: str = "F-score") -> pd.DataFrame:

    idx = data.groupby("Dataset")[metric].idxmax()
    bestDetectors = data.loc[idx].reset_index(drop=True)

    return bestDetectors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run MultiRocket on different datasets.')
    parser.add_argument('--benchmark', type=str, required=True, choices=datasets, help='Benchmark dataset to use (e.g., SMD, SMAP, MSL)')
    parser.add_argument('--metric', type = str, default = "F-score", help='Metric column used to select the best detector (default: F-score)')
    
    args = parser.parse_args()
    
    benchmark = args.benchmark
    metric = args.metric

    input_path = f"benchmarks/{benchmark}/analysis/benchmark_results.csv"
    output_path = f"benchmarks/{benchmark}/analysis/best_detectors_per_dataset.csv"

    data = pd.read_csv(input_path)

    results = chooseBestDetector(data, metric)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    results.to_csv(output_path,index = False)

    print(f"Saved {len(results)} entry to {output_path}")








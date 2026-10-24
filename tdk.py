import pandas as pd

FILE = "benchmarks/SMAP_MSL/analysis/benchmark_results.csv"

df = pd.read_csv(FILE)

metrics = [
    "Precision",
    "Recall",
    "F-score",
    "Accuracy",
    "MCC"
]

mean_metrics = (
    df.groupby("Algorithm")[metrics]
      .mean()
      .sort_values("F-score", ascending=False)
)

print("Átlagos metrikák algoritmusonként:")
print(mean_metrics)

print("\nLaTeX:")
print(
    mean_metrics.to_latex(
        float_format="%.4f",
        escape=False
    )
)
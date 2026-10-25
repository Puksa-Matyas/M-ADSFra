import pandas as pd
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pair the best performing detector to autoencoders output"
    )
    parser.add_argument("--fingerprint", type=str, required=True)
    parser.add_argument("--seed", type=int, required=True)

    args = parser.parse_args()
    fingerprint = args.fingerprint
    seed = args.seed

    tpot_predict = (
        f"benchmarks/SMAP_MSL/TPOT/output/"
        f"{fingerprint}_autoencoder_16_labeled_seed{seed}_predictions.csv"
    )
    benchmark_result = "benchmarks/SMAP_MSL/analysis/benchmark_results.csv"

    performance = pd.read_csv(benchmark_result)
    predicted = pd.read_csv(tpot_predict, index_col=0)

    result = predicted.merge(
        performance[["Dataset", "Algorithm", "F-score"]],
        left_on=["Dataset", "Predicted_Detector"],
        right_on=["Dataset", "Algorithm"],
        how="left",
        validate="many_to_one"
    )

    result = result.drop(columns=["Algorithm"])

    print(result[[
        "Dataset",
        "True_Detector",
        "Predicted_Detector",
        "F-score"
    ]])

    result.to_csv("tpot_with_f1_scores.csv", index=False)
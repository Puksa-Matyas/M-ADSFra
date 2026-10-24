import argparse
import pandas as pd
import os

algorithms = ['RZS','SPE','HST','TRAKR','TranAD']


def load_raw_fingerprints(input_folder: str, method: str) -> pd.DataFrame:
    """
    Ugyanaz a logika, mint a run_autoencoder.py load_fingerprints()
    függvényében: végigmegy input_folder összes fájlján, a fájlnévből
    (kiterjesztés és '_{method}' utótag levágásával) képzi a 'Dataset'
    oszlopot, és a nyers (nem autoencoderrel redukált) fingerprint
    feature-öket egyetlen DataFrame-be fűzi össze.
    """
    all_data = []

    filenames = sorted(os.listdir(input_folder))
    print(f"Found {len(filenames)} file(s) in {input_folder}")

    method_suffix = f'_{method}'

    for filename in filenames:
        file_path = os.path.join(input_folder, filename)

        if not os.path.isfile(file_path):
            continue

        dataset = os.path.splitext(filename)[0]
        if dataset.endswith(method_suffix):
            dataset = dataset[: -len(method_suffix)]

        data = pd.read_csv(file_path)

        if data.shape[0] == 0:
            # Nincs fejléc sor a fájlban -- az egyetlen adat-sort
            # fejlécként értelmezte a pandas. Újraolvassuk header nélkül.
            data = pd.read_csv(file_path, header=None)
            data.columns = [f'feature_{i}' for i in range(data.shape[1])]

        if 'Timestamp' in data.columns:
            data = data.drop(columns=['Timestamp'])

        data.insert(0, 'Dataset', dataset)
        all_data.append(data)
        print(f"  loaded {filename}  (shape={data.shape})")

    if not all_data:
        raise ValueError(f"No fingerprint files found in {input_folder}")

    return pd.concat(all_data, ignore_index=True)


def pairDetector(data_benchmark: pd.DataFrame, data_autoencoder: pd.DataFrame, metric: str = 'F-score', top_k: int = 3) -> pd.DataFrame:
    """
    Datasetenként megkeresi a legjobban teljesítő algoritmust
    (data_benchmark alapján, `metric` szerint), majd ennek az
    algoritmusnak a sorszámát (az `algorithms` listában elfoglalt
    indexét) hozzárendeli az autoencoder minden sorához, dataset
    szerint párosítva ('target' oszlop -- ez a TPOT tanítási célja).

    Emellett eltárolja a top-k legjobb algoritmus indexeit is egy
    külön, vesszővel elválasztott stringben ('Top{k}Targets' oszlop)
    -- ez CSAK a kiértékeléshez kell (top-k accuracy), a tanításhoz
    nem, mivel egy klasszifikátor egyetlen célváltozóra tanul.
    """
    algo_to_index = {name: i for i, name in enumerate(algorithms)}

    # Legjobb algoritmus datasetenként -- ez a tanítási target
    best_idx = data_benchmark.groupby('Dataset')[metric].idxmax()
    best = data_benchmark.loc[best_idx, ['Dataset', 'Algorithm']].reset_index(drop=True)
    best['target'] = best['Algorithm'].map(algo_to_index)

    unknown = best.loc[best['target'].isna(), 'Algorithm'].unique()
    if len(unknown) > 0:
        raise ValueError(
            f"Az alábbi algoritmusok nincsenek benne az `algorithms` listában: {list(unknown)}"
        )
    best['target'] = best['target'].astype(int)

    # Top-k legjobb algoritmus datasetenként -- csak a kiértékeléshez.
    # Az indexeket egy vesszővel elválasztott stringbe csomagoljuk,
    # mert egy CSV oszlop nem tud listát tárolni.
    def top_k_indices(group):
        top = group.sort_values(metric, ascending=False).head(top_k)['Algorithm']
        indices = [str(algo_to_index[a]) for a in top]
        return ','.join(indices)

    top_k_col_name = f'Top{top_k}Targets'
    top_k_series = data_benchmark.groupby('Dataset').apply(top_k_indices, include_groups=False)
    top_k_df = top_k_series.reset_index()
    top_k_df.columns = ['Dataset', top_k_col_name]

    best = best.merge(top_k_df, on='Dataset', how='left')

    # Párosítás Dataset szerint
    result = data_autoencoder.merge(
        best[['Dataset', 'target', top_k_col_name]],
        on='Dataset',
        how='left'
    )

    missing = result.loc[result['target'].isna(), 'Dataset'].unique()
    if len(missing) > 0:
        print(f"Warning: nincs benchmark eredmény ezekhez a datasetekhez: {list(missing)}")

    return result


if __name__ == "__main__":
    # Request the arguments
    parser = argparse.ArgumentParser(description='Pair the best performing detector to autoencoders output')
    parser.add_argument('--benchmark', type=str, required=True, help='Benchmark dataset to use (e.g., SMD, SMAP, MSL)')
    parser.add_argument('--method', type=str, required=True, help='Fingerprinting method / folder name to use (e.g., MultiRocket)')
    parser.add_argument('--n_components', type=int, default=4, help='Bottleneck dimension (default: 4)')
    parser.add_argument('--metric', type=str, default='F-score', help='Metric used to pick the best algorithm (default: F-score)')
    parser.add_argument('--top_k', type=int, default=3, help='Hány legjobb algoritmust tároljon el kiértékeléshez (default: 3)')
    # FIX: `type=bool` klasszikus argparse-csapda -- bool('False') is
    # True-t ad vissza (bármilyen nem-üres string igaz), tehát ez a
    # kapcsoló korábban MINDIG True lett volna, hívási módtól függetlenül.
    # store_true-val a kapcsoló jelenléte/hiánya dönt, alapból False.
    parser.add_argument('--skip_autoencoder', action='store_true',
                         help='Ha meg van adva, az autoencoderrel redukált fájl helyett a nyers '
                              'fingerprint fájlokat olvassa be és azokhoz rendeli a labelt.')

    # Parse the arguments
    args = parser.parse_args()

    # Use command-line arguments
    benchmark = args.benchmark
    method = args.method
    n_components = args.n_components

    benchmark_results = f"benchmarks/{benchmark}/analysis/benchmark_results.csv"

    print(f"Loading benchmark results from: {benchmark_results}")
    data_benchmark = pd.read_csv(benchmark_results)

    if args.skip_autoencoder:
        # Az autoencoder-lépés kihagyása: a nyers fingerprint fájlokat
        # olvassuk be közvetlenül a benchmarks/{benchmark}/fingerprint/{method}
        # mappából, ugyanúgy, mint a run_autoencoder.py tenné a redukció előtt.
        fingerprint_folder = f"benchmarks/{benchmark}/fingerprint/{method}"
        output_path = f"benchmarks/{benchmark}/TPOT/{method}_raw_labeled.csv"

        print(f"[skip_autoencoder] Loading raw fingerprints from: {fingerprint_folder}")
        data_features = load_raw_fingerprints(fingerprint_folder, method)
    else:
        autoencoder_results = f"benchmarks/{benchmark}/fingerprint/autoencoder/{method}_autoencoder_{n_components}.csv"
        output_path = f"benchmarks/{benchmark}/TPOT/{method}_autoencoder_{n_components}_labeled.csv"

        print(f"Loading autoencoder results from: {autoencoder_results}")
        data_features = pd.read_csv(autoencoder_results)

    result = pairDetector(data_benchmark, data_features, metric=args.metric, top_k=args.top_k)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    result.to_csv(output_path, index=False)
    print(f"Saved {len(result)} labeled rows to: {output_path}")

    # --- Detektor gyakoriság kiírása datasetenkénti bontásból ---
    index_to_algo = {i: name for i, name in enumerate(algorithms)}
    per_dataset_best = result.drop_duplicates('Dataset')['target'].map(index_to_algo)
    win_counts = per_dataset_best.value_counts().reindex(algorithms, fill_value=0)

    print("\nLegjobb detektor előfordulási gyakorisága:")
    for algo, count in win_counts.items():
        print(f"  {algo}: {count}")
import argparse
import os
import esig.backends as _esig_backends
import numpy as np
import pandas as pd
import roughpy as rp
from sklearn.decomposition import PCA
from sktime.transformations.signature_based import SignatureTransformer

datasets = ['MSL', 'SMAP', 'SMD', 'SMAP_MSL']

def _prepare_stream(self, stream_data, depth):
    no_samples, width = stream_data.shape
    increments = np.diff(stream_data, axis=0)
    indices = np.arange(no_samples - 1) / (no_samples - 1)
    context = rp.get_context(width, depth, rp.DPReal)
    return rp.LieIncrementStream.from_increments(increments, indices=indices, ctx=context)


_esig_backends.RoughPyBackend.prepare_stream = _prepare_stream

def load_series(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        arr = np.load(path)
    elif ext in ('.csv', '.txt'):
        arr = pd.read_csv(path).select_dtypes(include=[np.number]).to_numpy()
    else:
        raise ValueError(f"Unsupported file type: {path}")

    arr = np.asarray(arr, dtype=np.float64)
    return arr.reshape(-1, 1) if arr.ndim == 1 else arr


def reduce_channels(series, n_channels):
    n_components = min(n_channels, series.shape[0], series.shape[1])
    z = PCA(n_components=n_components).fit_transform(series)
    if n_components < n_channels:
        z = np.hstack([z, np.zeros((z.shape[0], n_channels - n_components))])
    return z

def strip_extension(benchmark, filename):
    return filename.replace('.txt' if benchmark == 'SMD' else '.npy', '')

def run_gsm(data, args):
    signature_transform = SignatureTransformer(
        augmentation_list=(args.augmentation,),
        window_name=args.window_type,
        window_depth=args.window_depth,
        window_length=args.window_size,
        window_step=args.window_step,
        rescaling=None if args.rescaling == 'none' else args.rescaling,
        sig_tfm=args.sig_tfm,
        depth=args.depth,
    )

    print(f"Raw data shape is: {data.shape}")
    fingerprint = signature_transform.fit_transform(data)
    print(f"Signature shape is: {fingerprint.shape}")

    return np.asarray(fingerprint, dtype=np.float64).reshape(-1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run fingerprinting on different datasets.')
    parser.add_argument('--benchmark', type=str, required=True, choices=datasets + ['all'], help='Benchmark dataset to use (e.g., SMD, SMAP, MSL, or all)')
    parser.add_argument('--augmentation', type=str, default='basepoint', choices=['addtime', 'leadlag', 'ir', 'cumsum', 'basepoint'])
    parser.add_argument('--window_type', type=str, default='global', choices=['global', 'expanding', 'sliding', 'dyadic'])
    parser.add_argument('--window_depth', type=int, default=3, help='Depth of the sliding window')
    parser.add_argument('--window_size', type=int, default=200, help='Size of the sliding window')
    parser.add_argument('--window_step', type=int, default=500, help='Step size of the sliding window')
    parser.add_argument('--rescaling', type=str, default='none', choices=['pre', 'post', 'none'])
    parser.add_argument('--sig_tfm', type=str, default='signature', choices=['signature', 'logsignature'])
    parser.add_argument('--depth', type=int, default=3, help='Depth parameter for the GSM algorithm')
    parser.add_argument('--n_channels', type=int, default=8, help='Number of PCA channels before the signature transform')
    parser.add_argument('--overwrite', action='store_true', help='Recompute fingerprints that already exist')

    args = parser.parse_args()

    benchmarks = datasets if args.benchmark == 'all' else [args.benchmark]

    for benchmark in benchmarks:
        print("Started Signature method on ", benchmark, "dataset.")
        folder_path = f'benchmarks/{benchmark}/data'
        out_path = f'benchmarks/{benchmark}/fingerprint/gsm'
        filename_list = os.listdir(f'{folder_path}/test')

        for i, filename in enumerate(filename_list, 1):
            out_filename = strip_extension(benchmark, filename) + '_gsm'
            out_file = os.path.join(out_path, out_filename + '.csv')

            print(f"[{i}/{len(filename_list)}] Generating fingerprint for: {filename}")
            if os.path.exists(out_file) and not args.overwrite:
                print(f"    Already exists: {out_filename}")
                continue

            try:
                series = load_series(f'{folder_path}/test/{filename}')  # (T, D)
            except ValueError as e:
                print(f"  Skipped: {e}")
                continue

            fingerprint = run_gsm(reduce_channels(series, args.n_channels).T[np.newaxis, :, :], args)

            os.makedirs(out_path, exist_ok=True)
            np.savetxt(
                out_file,
                fingerprint.reshape(1, -1),
                delimiter=',',
                fmt="%.8f",
                header=','.join(f'gsm_{j}' for j in range(fingerprint.shape[0])),
                comments='',
            )
            print("    Output feature vector size:", fingerprint.shape[0])
            print("    Saved fingerprint for:", out_filename)

        print(f"Done. Fingerprints saved to {out_path}")
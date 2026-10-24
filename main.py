import subprocess

# THE MAIN CONFIG FILE FOR THE FRAMEWORK

# Global
BENCHMARKS = ['SMAP', 'SMD']    # (SMD, SMAP, MSL)
METHODS = ['MultiRocket']       # MultiRocket
N_COMPONENTS = 4
DETECTORS = ['MDRS', 'TRAKR', 'SPE', 'HST', 'RZS', 'MTAD']      # ('MDRS', 'TRAKR', 'SPE', 'HST', 'RZS', 'MTAD')
SKIP_AUTOENCODER  = False

# --- MultiRocket grid ---------------------------------------------------
# N_KERNELS must always be a multiple of 84. Instead of a single fixed value
# we now sweep it: 1*84, 2*84, 3*84, ... up to N_KERNELS_MULTIPLIER_MAX.
# Edit N_KERNELS_MULTIPLIER_MAX (or replace the whole list) to control the range.
N_KERNELS_MULTIPLIER_MAX = 4  # -> tests 84, 168, 252, 336
#N_KERNELS_LIST = [84 * m for m in range(1, N_KERNELS_MULTIPLIER_MAX + 1)]
N_KERNELS_LIST = [84, 420, 840, 1680, 4200, 6300, 8400]
N_FEATURE_PER_KERNEL = 4

# --- TPOT grid ------------------------------------------------------------
# Every (generations, population_size) combination below is run against the
# SAME labeled data produced for a given N_KERNELS value.
TPOT_GENERATIONS_LIST = [5, 10, 20]
TPOT_POPULATION_SIZE_LIST = [30, 60, 100]
TPOT_CV_LIST = [3, 4, 5]  # now also a grid - edit freely, or set to a single-item list like [3]

# Where the single, ever-growing results table lives (created by run_TPOT.py)
TPOT_RESULTS_CSV = 'tpot_benchmark_results.csv'

STAGES_TO_RUN = [3, 4, 5, 6]
# Stages:
# 1 - Run detectors
# 2 - Evaluate detector results
# 3 - Fingerprinting
# 4 - Autoencoder
# 5 - Pair best detector
# 6 - TPOT


if __name__ == "__main__":

    for benchmark in BENCHMARKS:
        for method in METHODS:

            # 1. Detectors
            if 0 in STAGES_TO_RUN or 1 in STAGES_TO_RUN:
                ...

            # 2. Evaluate detector results
            if 0 in STAGES_TO_RUN or 2 in STAGES_TO_RUN:
                ...

            # From here on everything depends on N_KERNELS, so we loop over
            # the whole grid. Each iteration regenerates the fingerprints,
            # the autoencoder output and the labeled data from scratch
            # (overwriting the previous N_KERNELS' files - this is intentional).
            for n_kernels in N_KERNELS_LIST:

                print(f"\n{'#'*20} {benchmark} / {method} / n_kernels={n_kernels} {'#'*20}")

                # 3. Fingerprinting
                if 0 in STAGES_TO_RUN or 3 in STAGES_TO_RUN:
                    subprocess.run([
                        "py",
                        "run_fingerprinting.py",
                        "--benchmark", f"{benchmark}",
                        "--method", f"{method}",
                        "--n_kernels", f"{n_kernels}",
                        "--n_feature_per_kernel", f"{N_FEATURE_PER_KERNEL}",
                    ], check=True)

                # 4. Autoencoder
                if 0 in STAGES_TO_RUN or 4 in STAGES_TO_RUN:
                    if not SKIP_AUTOENCODER:
                        subprocess.run([
                            r"autoencoder\Scripts\python.exe",
                            "run_autoencoder.py",
                            "--benchmark", f"{benchmark}",
                            "--method", f"{method}",
                            "--n_components", f"{N_COMPONENTS}",
                        ], check=True)

                # 5. Pair best detector
                if 0 in STAGES_TO_RUN or 5 in STAGES_TO_RUN:
                    if not SKIP_AUTOENCODER:
                        subprocess.run([
                            "py",
                            "run_pair_best_detector.py",
                            "--benchmark", f"{benchmark}",
                            "--method", f"{method}",
                            "--n_components", f"{N_COMPONENTS}",
                        ], check=True)
                    else:
                        subprocess.run([
                            "py",
                            "run_pair_best_detector.py",
                            "--benchmark", f"{benchmark}",
                            "--method", f"{method}",
                            "--n_components", f"{N_COMPONENTS}",
                            "--skip_autoencoder",f"{SKIP_AUTOENCODER}"
                            ], check=True)


                # 6. TPOT - sweep generations x population_size x cv for THIS n_kernels
                if 0 in STAGES_TO_RUN or 6 in STAGES_TO_RUN:
                    for generations in TPOT_GENERATIONS_LIST:
                        for population_size in TPOT_POPULATION_SIZE_LIST:
                            for cv in TPOT_CV_LIST:

                                print(f"  -> TPOT run: generations={generations}, "
                                      f"population_size={population_size}, cv={cv}")

                                subprocess.run([
                                    r"TPOT\Scripts\python.exe",
                                    "run_TPOT.py",
                                    "--benchmark", f"{benchmark}",
                                    "--method", f"{method}",
                                    "--n_components", f"{N_COMPONENTS}",
                                    "--n_kernels", f"{n_kernels}",
                                    "--generations", f"{generations}",
                                    "--population_size", f"{population_size}",
                                    "--cv", f"{cv}",
                                    "--results_csv", TPOT_RESULTS_CSV,
                                ], check=True)

    print(f"\nAll done. Full benchmark grid results appended to: {TPOT_RESULTS_CSV}")
import numpy as np
import time
import os
import argparse
import csv
import matplotlib.pyplot as plt
import seaborn as sns
from numba import njit, prange, set_num_threads
from sklearn.feature_extraction.image import extract_patches_2d
from skimage import io, color


import numpy as np
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
from numba import set_num_threads

# --- NUMBA KERNELS (Optimized for Parallelism) ---
from numba_ak_svd import OMP, kSVD


# --- DATA & BENCHMARKING ---

def load_bsds500(path, num_images=10, patch_size=(8, 8)):
    all_patches = []
    files = [f for f in os.listdir(path) if f.endswith('.jpg')][:num_images]
    
    for f in files:
        img = io.imread(os.path.join(path, f))
        if img.ndim == 3:
            img = color.rgb2gray(img)
        # Extracting many patches to ensure N is large enough for parallelism
        patches = extract_patches_2d(img, patch_size, max_patches=2000)
        all_patches.append(patches.reshape(len(patches), -1))
        
    Y = np.vstack(all_patches).T.astype(np.float32)
    # Zero-center patches
    Y -= np.mean(Y, axis=0)
    return Y

def run_omp_benchmark(Y, D, T_0, threads_list, batch_size=1):
    omp_results = []
    output_dir = "/omp_results"
    os.makedirs(output_dir, exist_ok=True)

    if D is None:
        # Initialize a random dictionary if not provided
        k = 256  # Number of atoms
        rng = np.random.default_rng(seed=42)
        D = rng.standard_normal(size=(k, Y.shape[1]), dtype=np.float32)
        D /= np.linalg.norm(D, ord=2, axis=1, keepdims=True)

    for t in threads_list:
        set_num_threads(t)
        
        # warm-up run (not timed)
        _ = OMP(Y, T_0=T_0, D=D, batch_size=batch_size)

        # We run multiple trials to get a distribution for the histogram
        trial_times = []
        for trial in range(5): 
            start = time.perf_counter()
            X = OMP(Y, T_0=T_0, D=D, batch_size=batch_size)
            trial_times.append(time.perf_counter() - start)
        
        avg_time = np.mean(trial_times)
        omp_results.append({
            "threads": t,
            "avg_runtime": avg_time,
            "all_trials": trial_times
        })
        print(f"OMP {t} threads: {avg_time:.4f}s")
#    # Save results to CSV
    df = pd.DataFrame(omp_results)
    df.to_csv(f"{output_dir}/omp_scaling_N{Y.shape[1]}.csv", index=False)

    return omp_results    
# --- PLOTTING ---

def generate_visuals(results):
    sns.set_theme(style="whitegrid")
    threads = [r['threads'] for r in results]
    runtimes = [r['runtime'] for r in results]
    losses = [r['loss'] for r in results]

    fig, ax1 = plt.subplots(1, 2, figsize=(14, 5))

    # Loss vs Runtime
    sns.scatterplot(x=runtimes, y=losses, hue=threads, ax=ax1[0], palette="viridis", s=100)
    ax1[0].set_title("Reconstruction Loss vs. Runtime")
    ax1[0].set_xlabel("Runtime (seconds)")
    ax1[0].set_ylabel("RMSE Loss")

    # Runtime Histogram/Bar
    sns.barplot(x=threads, y=runtimes, ax=ax1[1], palette="magma")
    ax1[1].set_title("Runtime scaling by Thread Count")
    ax1[1].set_xlabel("Number of Threads")
    ax1[1].set_ylabel("Time (seconds)")

    plt.tight_layout()
    plt.savefig("ksvd_benchmark_results.png")
    print("Plots saved as ksvd_benchmark_results.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Path to BSDS500 images")
    parser.add_argument("--images", type=int, default=20)
    args = parser.parse_args()

    # Load and Preprocess
    print("Loading BSDS500 patches...")
    Y_data = load_bsds500(args.data_path, num_images=args.images)
    print(f"Dataset shape: {Y_data.shape} (Features x Samples)")

    # Benchmark config
    thread_counts = [1, 2, 4, 8, 12, 16] # Adjust based on laptop/TACC core count
    benchmark_data = run_omp_benchmark(Y_data, D=None, T_0=32, threads_list=thread_counts)
    
    generate_visuals(benchmark_data)
import numba
import numpy as np
import time
import os
import argparse
import seaborn as sns
from numba import njit, prange, set_num_threads
from sklearn.feature_extraction.image import extract_patches_2d
from skimage import io, color
import pandas as pd
import matplotlib.pyplot as plt

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

def run_ksvd_benchmark(Y, T_0, k, num_iter, threads_list, batch_size=1):
    results = []
    output_dir = "ksvd_results"
    os.makedirs(output_dir, exist_ok=True)

    # warmup run (not timed)
    _ = kSVD(Y, T_0=T_0, k=k, num_iter=1, batch_size=batch_size, track_loss=False, verbose=0)

    for t in threads_list:
        set_num_threads(t)
        print(f"--- Running kSVD with {t} Threads ---")
        
        start_time = time.perf_counter()
        # Using your specific kSVD signature
        # D_final, X_final (and potentially loss_history)
        D_out, X_out, loss = kSVD(
            Y, T_0=T_0, k=k, num_iter=num_iter, 
            batch_size=batch_size, track_loss=True, verbose=0
        )
        end_time = time.perf_counter()
        
        duration = end_time - start_time
        
        # Final Reconstruction Loss
        print(D_out.shape, X_out.shape)
        reconstruction = X_out @ D_out  # Shape: (n_samples, n_features)
        final_loss = np.sqrt(np.mean((Y - reconstruction)**2))
        
        results.append({
            "threads": t,
            "runtime": duration,
            "loss": final_loss,
            "throughput": Y.shape[1] / duration # Samples per second
        })

    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv(f"{output_dir}/ksvd_scaling_N{Y.shape[1]}.csv", index=False)
    return df
    
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


def plot_results(df, mode="ksvd"):
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    
    if mode == "ksvd":
        # 1. Loss vs Runtime
        ax[0].scatter(df['runtime'], df['loss'], c=df['threads'], cmap='viridis', s=100)
        for i, txt in enumerate(df['threads']):
            ax[0].annotate(f"{txt}T", (df['runtime'][i], df['loss'][i]))
        ax[0].set_title("Reconstruction Quality vs. Wall Clock Time")
        ax[0].set_xlabel("Total Runtime (s)")
        ax[0].set_ylabel("RMSE Loss")

        # 2. Speedup Plot
        base_time = df[df['threads'] == 1]['runtime'].values[0]
        ax[1].plot(df['threads'], base_time / df['runtime'], marker='o', label="Observed Speedup")
        ax[1].plot(df['threads'], df['threads'], linestyle='--', color='red', label="Ideal Linear")
        ax[1].set_title("Parallel Speedup (Strong Scaling)")
        ax[1].set_xlabel("Thread Count")
        ax[1].legend()

    plt.tight_layout()
    plt.savefig(f"benchmarks/{mode}_performance.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True, help="Path to BSDS500 images")
    parser.add_argument("--images", type=int, default=20)
    args = parser.parse_args()

    # Load and Preprocess
    print("Loading BSDS500 patches...")
    patch_size = (7,7)
    Y_data = load_bsds500(args.data_path, num_images=args.images, patch_size=patch_size)

    print(Y_data.shape)
    Y_data = Y_data.astype(np.float32).T  # Shape: (features, samples)
    print(f"Dataset shape: {Y_data.shape} (Features x Samples)")

    # Benchmark config
    thread_counts = []
    for i in range(numba.config.NUMBA_NUM_THREADS):
        if 2**i <= numba.config.NUMBA_NUM_THREADS:
            thread_counts.append(2**i)
    benchmark_data = run_ksvd_benchmark(Y_data, T_0=32, k=256, num_iter=10, threads_list=thread_counts, batch_size=128)

    # generate_visuals(benchmark_data)
import numpy as np
import numba
import time
import os
import re
import sys
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from numba import set_num_threads

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from Numba_Approx_V2 import kSVD as kSVD_numba_seq
from Para_Numba_Approx_V1 import kSVD as kSVD_numba_par
from Serial_V2_1 import kSVD as kSVD_serial

CPP_BINARY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../Implementations/AK-SVD-Parallel/build/main"
)


def make_synthetic_data(n_samples=400000, n_features=49):
    rng = np.random.default_rng(0)
    Y = rng.normal(loc=5.0, scale=2.0, size=(n_samples, n_features)).astype(np.float32)
    Y -= np.mean(Y, axis=0)
    Y /= (np.std(Y, axis=0) + 1e-8)
    return Y


def run_cpp(n_samples, n_features, threads):
    """Run the C++ AK-SVD-Parallel binary and parse its timing output.
    Note: the binary hardcodes K=256, T0=32, batch_size=128, num_iter=2.
    """
    cmd = [CPP_BINARY, "--threads", str(threads), "--synth", str(n_samples), str(n_features)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    out = result.stdout
    mean_match = re.search(r"mean\s*:\s*([\d.]+)", out)
    loss_match  = re.search(r"loss\s*:\s*([\d.]+)", out)
    if not mean_match:
        raise RuntimeError(f"Could not parse C++ output:\n{out}\n{result.stderr}")
    return float(mean_match.group(1)), float(loss_match.group(1)) if loss_match else float("nan")


def run_benchmark(Y, T_0, k, num_iter, batch_size, threads_list):
    results = []
    n_samples, n_features = Y.shape
    output_dir = "ksvd_results"
    os.makedirs(output_dir, exist_ok=True)

    # --- Warmup numba JIT (both variants) ---
    print("Warming up Numba JIT...")
    Y_warm = Y[:1000]
    set_num_threads(1)
    kSVD_numba_seq(Y_warm, T_0=T_0, k=k, num_iter=1, batch_size=batch_size, verbose=0)
    set_num_threads(12)
    kSVD_numba_par(Y_warm, T_0=T_0, k=k, num_iter=1, batch_size=batch_size, verbose=0)
    print("Warmup done.\n")

    # --- Serial baseline ---
    print("Running serial kSVD...")
    t0 = time.perf_counter()
    D_s, loss_s = kSVD_serial(Y, T_0=T_0, k=k, num_iter=num_iter, verbose=0)
    serial_time = time.perf_counter() - t0
    print(f"  Serial time: {serial_time:.2f}s  final loss: {loss_s[-1]:.4f}\n")
    results.append({
        "impl": "serial",
        "threads": 1,
        "runtime": serial_time,
        "final_loss": float(loss_s[-1]),
        "throughput": n_samples / serial_time,
    })

    for t in threads_list:
        set_num_threads(t)

        # --- Numba sequential OMP ---
        print(f"Running Numba-seq kSVD with {t} thread(s)...")
        t0 = time.perf_counter()
        D_ns, loss_ns = kSVD_numba_seq(Y, T_0=T_0, k=k, num_iter=num_iter,
                                        batch_size=batch_size, verbose=0)
        elapsed = time.perf_counter() - t0
        print(f"  Numba-seq ({t}T): {elapsed:.2f}s  final loss: {loss_ns[-1]:.4f}")
        results.append({
            "impl": f"numba_seq_{t}T",
            "threads": t,
            "runtime": elapsed,
            "final_loss": float(loss_ns[-1]),
            "throughput": n_samples / elapsed,
        })

        # --- Numba parallel OMP ---
        print(f"Running Numba-par kSVD with {t} thread(s)...")
        t0 = time.perf_counter()
        D_np, loss_np = kSVD_numba_par(Y, T_0=T_0, k=k, num_iter=num_iter,
                                        batch_size=batch_size, verbose=0)
        elapsed = time.perf_counter() - t0
        print(f"  Numba-par ({t}T): {elapsed:.2f}s  final loss: {loss_np[-1]:.4f}")
        results.append({
            "impl": f"numba_par_{t}T",
            "threads": t,
            "runtime": elapsed,
            "final_loss": float(loss_np[-1]),
            "throughput": n_samples / elapsed,
        })

        # --- C++ AK-SVD-Parallel (K=256, T0=32, batch_size=128, num_iter=2 hardcoded) ---
        print(f"Running C++ AK-SVD-Parallel with {t} thread(s)...")
        cpp_time, cpp_loss = run_cpp(n_samples, n_features, t)
        print(f"  C++ ({t}T): {cpp_time:.2f}s  final loss: {cpp_loss:.4f}")
        results.append({
            "impl": f"cpp_{t}T",
            "threads": t,
            "runtime": cpp_time,
            "final_loss": cpp_loss,
            "throughput": n_samples / cpp_time,
        })

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, f"ksvd_comparison_N{n_samples}_synthetic.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")
    return df


def plot_results(df, output_dir="ksvd_results"):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style="whitegrid")

    labels = df["impl"].tolist()
    runtimes = df["runtime"].tolist()

    def color_for(label):
        if "serial" in label:   return "steelblue"
        if "numba_seq" in label: return "mediumseagreen"
        if "numba_par" in label: return "tomato"
        if "cpp" in label:       return "goldenrod"
        return "gray"

    palette = [color_for(l) for l in labels]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    bars = axes[0].bar(labels, runtimes, color=palette)
    for bar, rt in zip(bars, runtimes):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"{rt:.2f}s", ha="center", va="bottom", fontsize=8)
    axes[0].set_title("Runtime Comparison")
    axes[0].set_xlabel("Implementation")
    axes[0].set_ylabel("Wall-clock time (s)")
    axes[0].tick_params(axis="x", rotation=30)

    serial_time = df[df["impl"] == "serial"]["runtime"].values[0]
    non_serial = df[df["impl"] != "serial"].copy()
    if not non_serial.empty:
        for impl_label, group in non_serial.groupby(
                non_serial["impl"].str.replace(r"_\d+T$", "", regex=True)):
            speedups = serial_time / group["runtime"].values
            axes[1].plot(group["threads"].values, speedups, marker="o", label=impl_label)
        axes[1].plot(non_serial["threads"].values,
                     non_serial["threads"].values,
                     linestyle="--", color="black", alpha=0.4, label="Ideal linear")
        axes[1].set_title("Speedup vs Serial")
        axes[1].set_xlabel("Thread count")
        axes[1].set_ylabel("Speedup")
        axes[1].legend()

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "ksvd_comparison_synthetic.png")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    Y_data = make_synthetic_data(n_samples=100000, n_features=49)
    print(f"Dataset shape: {Y_data.shape}  (n_samples x n_features)\n")

    max_threads = numba.config.NUMBA_NUM_THREADS
    print("max_threads:", max_threads)
    thread_counts = [12]

    df = run_benchmark(
        Y_data,
        T_0=5,
        k=256,
        num_iter=2,
        batch_size=256,
        threads_list=thread_counts,
    )

    print("\n--- Summary ---")
    print(df.to_string(index=False))

    plot_results(df)

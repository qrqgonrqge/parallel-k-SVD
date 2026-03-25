import numpy as np
import timeit
import csv
import numba
from numba import set_num_threads
from sklearn.datasets import make_regression

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# -----------------------------
# IMPORT YOUR FUNCTIONS
# -----------------------------
from Experiments.Numba.OMP_Implementations.OMP_numba_parallel import (
    OMP_serial,
    OMP_numba_serial,
    OMP_numba_jit,
    OMP_numba_njit,
    OMP_numba_njit_parallel,
    OMP_numba_njit_parallel_pbatches,
)

# -----------------------------
# DATA SETUP
# -----------------------------
D, y = make_regression(
    n_samples=50,
    n_features=300,
    n_targets=20000,
    noise=4,
    random_state=0
)

Y = y.T
D = D.T

n_samples = Y.shape[0]

# -----------------------------
# PARAMS
# -----------------------------
T_0 = 1
N = 10  # timeit runs

functions = [
    ("OMP_serial", OMP_serial),
    ("OMP_numba_serial", OMP_numba_serial),
    ("OMP_numba_jit", OMP_numba_jit),
    ("OMP_numba_njit", OMP_numba_njit),
    # ("OMP_numba_njit_parallel", OMP_numba_njit_parallel),
    ("OMP_numba_njit_parallel_pbatches", OMP_numba_njit_parallel_pbatches),
]

max_threads = numba.get_num_threads()

thread_list = [2**i for i in range(int(np.log2(max_threads)) + 1)]
batch_sizes = [2**i for i in range(int(np.log2(n_samples)) + 1)]
batch_sizes = [b for b in batch_sizes if b <= n_samples]

print("Threads:", thread_list)
print("Batch sizes:", batch_sizes)

# # -----------------------------
# # WARMUP (compile all functions)
# # -----------------------------
# for name, fn in functions:
#     try:
#         fn(Y, T_0, D, batch_size=1)
#     except:
#         pass

# -----------------------------
# BENCHMARK
# -----------------------------
results = []

for name, fn in functions:
    print(f"\n--- {name} ---")
    
    for n_threads in thread_list:
        set_num_threads(n_threads)
        
        for batch_size in batch_sizes:
            # define callable for timeit
            stmt = lambda: fn(Y, T_0, D, batch_size=batch_size)
            
            try:
                stmt()
            except:
                print(f"FAIL: {name}, threads={n_threads}, batch={batch_size}")
                continue
            try:
                t = timeit.timeit(stmt, number=N) / N
            except Exception as e:
                print(f"FAIL: {name}, threads={n_threads}, batch={batch_size}")
                continue

            print(f"{name} | threads={n_threads} | batch={batch_size} | {t:.6f}s")

            results.append((name, n_threads, batch_size, t))

# -----------------------------
# SAVE CSV
# -----------------------------
with open("omp_benchmark_all.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["function", "threads", "batch_size", "time_sec"])
    writer.writerows(results)

print("\nSaved to omp_benchmark_all.csv")
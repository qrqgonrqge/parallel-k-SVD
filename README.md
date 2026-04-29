# parallel-k-SVD
Contributors: Varun Srinivasan, Raymond Jiang, Richard Nguyen.

Implementation and benchmarking of a parallel K-SVD dictionary learning algorithm, with Python (Numba) and C++ (Eigen + TBB/OpenMP) backends.

---

## Repository layout

```
Implementations/
  AK_SVD/            Python approximate K-SVD variants
  K_SVD/             Python exact K-SVD serial variants
  AK-SVD-Serial/     C++ approximate K-SVD, single-threaded
  AK-SVD-Parallel/   C++ approximate K-SVD, TBB parallel
  K-SVD-Parallel/    C++ exact K-SVD, OpenMP parallel
  K-SVD-Serial-V4/   C++ exact K-SVD, single-threaded
comparison_folder/   Benchmark scripts (synthetic & image data)
```

---

## Python installation

### Requirements
- Python 3.9+
- pip

### Steps

```bash
# 1. Clone the repo
git clone https://github.com/qrqgonrqge/parallel-k-SVD.git
cd parallel-k-SVD

# 2. (Recommended) create a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

The `requirements.txt` installs:

| Package | Purpose |
|---|---|
| `numpy` | Array math |
| `scipy` | SVD, sparse solvers |
| `scikit-learn` | OMP (`orthogonal_mp_gram`) |
| `numba` | JIT compilation / parallelism |
| `matplotlib` | Plotting |
| `pandas`, `seaborn` | Benchmark result tables / plots |

### Running a benchmark

```bash
cd comparison_folder
python benchmark_ksvd_synthetic.py
```

Results are written to `comparison_folder/ksvd_results/`.

---

## C++ installation

### System dependencies

Install the following via your package manager before building.

**Ubuntu / Debian**
```bash
sudo apt update
sudo apt install cmake build-essential libeigen3-dev libopencv-dev libtbb-dev
```



### Building each C++ implementation

All four C++ projects follow the same CMake workflow.
Replace `<impl>` with the directory name you want to build.

```bash
cd Implementations/<impl>
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

| Directory | Parallelism | Extra dep |
|---|---|---|
| `AK-SVD-Serial` | none | Eigen, OpenCV |
| `AK-SVD-Parallel` | Intel TBB | Eigen, OpenCV, TBB |
| `K-SVD-Parallel` | OpenMP | Eigen, OpenCV |
| `K-SVD-Serial-V4` | none | Eigen, OpenCV |

The compiled binary is placed at `Implementations/<impl>/build/main`.

### Running a C++ binary

**Image mode** (default — requires `images/` directory next to the binary):
```bash
cd Implementations/AK-SVD-Parallel
./build/main
```

**Synthetic data mode** (no image file needed):
```bash
# --synth <n_samples> [<n_features=49>]
./build/main --synth 100000 49

# Control thread count (AK-SVD-Parallel only)
./build/main --threads 8 --synth 100000 49
```

Hardcoded hyperparameters (same across all C++ builds): `K=256`, `T0=32`, `batch_size=128`, `num_iter=2`.

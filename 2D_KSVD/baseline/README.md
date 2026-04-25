# 2D K-SVD Baseline Implementation

This directory contains implementations of the 2D K-SVD algorithm converted from the original Jupyter notebook.

## Files

- `2d_baseline.ipynb` - Original Python implementation
- `2d_baseline.cpp` - C++ implementation using Eigen library
- `2d_baseline_cuda.cu` - CUDA implementation for GPU acceleration
- `CMakeLists.txt` - Build configuration for C++ version
- `CMakeLists_cuda.txt` - Build configuration for CUDA version

## Dependencies

### C++ Version
- C++17 compiler
- Eigen3 library
- CMake 3.10+

### CUDA Version
- CUDA toolkit 11.0+
- CUBLAS
- CUSOLVER
- CMake 3.10+

## Building

### C++ Version
```bash
mkdir build_cpp
cd build_cpp
cmake -f ../CMakeLists.txt ..
make
./2d_baseline
```

### CUDA Version
```bash
mkdir build_cuda
cd build_cuda
cmake -f ../CMakeLists_cuda.txt ..
make
./2d_baseline_cuda
```

## Implementation Notes

### C++ Version
- Uses Eigen library for linear algebra operations
- Implements the same algorithm structure as the Python version
- Includes all key functions: normalize_columns, generate_synthetic, omp, ksvd

### CUDA Version
- Implements GPU-accelerated versions of key operations
- Uses CUBLAS for matrix operations
- Uses CUSOLVER for SVD operations
- Includes CUDA kernels for parallel computation
- Note: The CUDA version is a simplified implementation that would need additional work for full optimization

## Algorithm Overview

The K-SVD algorithm consists of two main phases:

1. **Sparse Coding Phase**: Uses Orthogonal Matching Pursuit (OMP) to find sparse representations
2. **Dictionary Update Phase**: Updates dictionary atoms using SVD decomposition

The algorithm iteratively improves the dictionary to better represent the input data with sparse coefficients.

## Performance Considerations

- The C++ version provides a good baseline and should be faster than the Python version
- The CUDA version can provide significant speedup for large datasets but requires CUDA-capable GPU
- For production use, consider further optimizations like memory coalescing, shared memory usage, and kernel fusion

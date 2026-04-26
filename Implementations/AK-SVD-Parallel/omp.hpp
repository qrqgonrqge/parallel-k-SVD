#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/SVD>

class OMP {

    public:
    struct Params {
        int t0         = 0;
        int K          = 0;
        int N          = 0;
        int batch_size = 256;
    };
    OMP(int N, int K, int T0, Eigen::MatrixXf Y, int batch_size = 256);
    Params params;
    Eigen::MatrixXf Y;

    bool do_omp();

    Eigen::MatrixXf D;  // N x K: each column is a dictionary atom
    Eigen::MatrixXf X;  // M x K: sparse coefficients (output)
};


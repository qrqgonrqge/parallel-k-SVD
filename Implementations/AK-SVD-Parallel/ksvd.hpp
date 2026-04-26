#pragma once
#include <Eigen/Dense>
#include <Eigen/SVD>
#include "omp.hpp"

class KSVD {
public:
    KSVD(int K, int T0, int num_iter, int batch_size = 256, unsigned int seed = 42);

    // Learn dictionary D from data matrix Y (M signals × N dims).
    void fit(const Eigen::MatrixXf& Y);

    Eigen::MatrixXf D;     // N × K  — each column is one dictionary atom
    Eigen::VectorXf loss;  // Frobenius reconstruction error per iteration

private:
    int K_, T0_, num_iter_, batch_size_;
    unsigned int seed_;
};

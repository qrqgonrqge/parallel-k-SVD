#include "omp.hpp"

OMP::OMP(int N, int K, int T0, Eigen::MatrixXd Y) {
    this->params.N = N;
    this->params.K = K;
    this->params.t0 = T0;
    this->Y = Y;
}
bool OMP::do_omp() {
    printf("Staring OMP...\n");
    
    // N: dimensionality of signal
    const int N = static_cast<int>(this->Y.cols());

    // M: number of signals
    const int M = static_cast<int>(this->Y.rows());

    // K: number of atoms
    const int K = this->params.K;

    Eigen::MatrixXd X = Eigen::MatrixXd::Zero(K, M);

    Eigen::VectorXd D_norm(K);
    for (int k = 0; k < K; k++) {
        D_norm(k) = this->D.col(k).norm();
    }

    std::vector<int> selected;
    Eigen::MatrixXd A(this->params.t0, M);
    for (int m = 0; m < M; m++) {
        
    }


    return true;
}

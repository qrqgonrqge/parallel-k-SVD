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
        // T0: sparsity level
        int t0 = 0;
        int K = 0;
        int N = 0;
    };
    OMP(int N, int K, int T0, Eigen::MatrixXd Y);
    Params params;
    Eigen::MatrixXd Y;

    bool do_omp();
};


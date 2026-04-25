#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <Eigen/Dense>
#include <Eigen/SVD>

using namespace Eigen;
using namespace std;

// Functions
MatrixXd normalize_columns(const MatrixXd& D) {
    MatrixXd result = D;
    for (int i = 0; i < D.cols(); ++i) {
        double norm = D.col(i).norm();
        if (norm > 1e-12) {
            result.col(i) /= norm;
        }
    }
    return result;
}

// m: number of rows in X
// K: number of atoms in dictionary
// N: number of columns in X, dimension of signal
// T0: sparsity level
tuple<MatrixXd, MatrixXd, MatrixXd> generate_synthetic(int m = 20, int K = 40, int N = 200, int T0 = 3, int seed = 0) {
    mt19937 gen(seed);
    normal_distribution<double> dist(0.0, 1.0);
    
    // Generate true dictionary
    MatrixXd D_true = MatrixXd::NullaryExpr(m, K, [&](int, int) { return dist(gen); });
    D_true = normalize_columns(D_true);
    
    // Generate true coefficients
    MatrixXd C_true = MatrixXd::Zero(K, N);
    for (int i = 0; i < N; ++i) {
        vector<int> indices(K);
        iota(indices.begin(), indices.end(), 0);
        shuffle(indices.begin(), indices.end(), gen);
        
        for (int j = 0; j < T0; ++j) {
            C_true(indices[j], i) = dist(gen);
        }
    }
    
    // Generate data
    MatrixXd X = D_true * C_true;
    
    return make_tuple(D_true, C_true, X);
}

VectorXd omp(const MatrixXd& D, const VectorXd& x, int T0) {
    int m = D.rows();
    int K = D.cols();
    
    if (x.size() != m) {
        cout << "Dimension mismatch: D has " << m << " rows, x has " << x.size() << " elements" << endl;
        return VectorXd::Zero(K);
    }
    
    VectorXd residual = x;
    vector<int> support;
    
    for (int iter = 0; iter < T0; ++iter) {
        VectorXd correlations = D.transpose() * residual;
        int idx;
        correlations.cwiseAbs().maxCoeff(&idx);
        
        if (find(support.begin(), support.end(), idx) != support.end()) {
            break;
        }
        support.push_back(idx);
        
        MatrixXd Ds(m, support.size());
        for (int i = 0; i < support.size(); ++i) {
            Ds.col(i) = D.col(support[i]);
        }
        
        if (Ds.cols() > 0 && Ds.rows() > 0) {
            VectorXd c_s = Ds.jacobiSvd(ComputeThinU | ComputeThinV).solve(x);
            residual = x - Ds * c_s;
        }
        
        if (residual.norm() < 1e-6) {
            break;
        }
    }
    
    VectorXd c = VectorXd::Zero(K);
    if (!support.empty()) {
        MatrixXd Ds(m, support.size());
        for (int i = 0; i < support.size(); ++i) {
            Ds.col(i) = D.col(support[i]);
        }
        VectorXd c_s = Ds.jacobiSvd(ComputeThinU | ComputeThinV).solve(x);
        for (int i = 0; i < support.size(); ++i) {
            if (support[i] >= 0 && support[i] < K) {
                c(support[i]) = c_s(i);
            }
        }
    }
    
    return c;
}

vector<double> errors;

pair<MatrixXd, MatrixXd> ksvd(const MatrixXd& X, int K, int T0, int n_iter = 10) {
    int m = X.rows();
    int N = X.cols();
    
    // Initialize dictionary randomly from data
    vector<int> indices(N);
    iota(indices.begin(), indices.end(), 0);
    random_shuffle(indices.begin(), indices.end());
    
    MatrixXd D(m, K);
    for (int i = 0; i < K; ++i) {
        D.col(i) = X.col(indices[i]);
    }
    D = normalize_columns(D);
    
    MatrixXd C = MatrixXd::Zero(K, N);
    
    for (int it = 0; it < n_iter; ++it) {
        cout << "Iteration " << it + 1 << endl;
        
        // ---- Sparse Coding ----
        for (int i = 0; i < N; ++i) {
            if (i % 50 == 0) {
                cout << "Processing signal " << i << "/" << N << endl;
            }
            VectorXd c_coeffs = omp(D, X.col(i), T0);
            if (c_coeffs.size() != K) {
                cout << "OMP returned wrong size: " << c_coeffs.size() << " expected: " << K << endl;
                return make_pair(MatrixXd::Zero(m, K), MatrixXd::Zero(K, N));
            }
            C.col(i) = c_coeffs;
        }
        
        // ---- Dictionary Update ----
        for (int k = 0; k < K; ++k) {
            vector<int> idx;
            for (int i = 0; i < N; ++i) {
                if (C(k, i) != 0) {
                    idx.push_back(i);
                }
            }
            
            if (idx.empty()) {
                continue;
            }
            
            // Compute residual excluding atom k
            MatrixXd E(m, idx.size());
            for (int i = 0; i < idx.size(); ++i) {
                if (idx[i] >= 0 && idx[i] < N) {
                    E.col(i) = X.col(idx[i]) - D * C.col(idx[i]) + D.col(k) * C(k, idx[i]);
                }
            }
            
            // Only perform SVD if we have enough data
            if (E.cols() > 0 && E.rows() > 0) {
                try {
                    // Rank-1 SVD
                    JacobiSVD<MatrixXd> svd(E, ComputeThinU | ComputeThinV);
                    if (svd.matrixU().cols() > 0 && svd.matrixV().rows() > 0 && svd.singularValues().size() > 0) {
                        D.col(k) = svd.matrixU().col(0);
                        
                        for (int i = 0; i < idx.size(); ++i) {
                            if (idx[i] >= 0 && idx[i] < N && i < svd.matrixV().cols()) {
                                C(k, idx[i]) = svd.singularValues()(0) * svd.matrixV()(0, i);
                            }
                        }
                    }
                } catch (...) {
                    cout << "SVD failed for atom " << k << endl;
                    continue;
                }
            }
        }
        
        D = normalize_columns(D);
        
        // Track reconstruction error
        double error = (X - D * C).norm() / X.norm();
        cout << "Reconstruction error: " << error << endl;
        errors.push_back(error);
    }
    
    return make_pair(D, C);
}

// Validation
double reconstruction_error(const MatrixXd& X, const MatrixXd& D, const MatrixXd& C) {
    return (X - D * C).norm() / X.norm();
}

double dictionary_recovery_score(const MatrixXd& D_true, const MatrixXd& D_learned) {
    MatrixXd sims = (D_true.transpose() * D_learned).cwiseAbs();
    double total = 0.0;
    for (int i = 0; i < sims.rows(); ++i) {
        total += sims.row(i).maxCoeff();
    }
    return total / sims.rows();
}

int main() {
    int m = 20, K = 40, N = 200, T0 = 3;
    
    cout << "Generating synthetic data..." << endl;
    auto [D_true, C_true, X] = generate_synthetic(m, K, N, T0);
    cout << "D_true: " << D_true.rows() << "x" << D_true.cols() << endl;
    cout << "X: " << X.rows() << "x" << X.cols() << endl;
    
    cout << "Starting KSVD..." << endl;
    auto [D_learned, C_learned] = ksvd(X, K, T0, 10);
    
    cout << "\nFinal Reconstruction Error: " << reconstruction_error(X, D_learned, C_learned) << endl;
    cout << "Dictionary Recovery Score: " << dictionary_recovery_score(D_true, D_learned) << endl;
    
    return 0;
}

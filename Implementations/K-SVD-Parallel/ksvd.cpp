#include "ksvd.hpp"
#include <random>
#include <vector>

KSVD::KSVD(int K, int T0, int num_iter, int batch_size, unsigned int seed)
    : K_(K), T0_(T0), num_iter_(num_iter), batch_size_(batch_size), seed_(seed) {}

void KSVD::fit(const Eigen::MatrixXd& Y) {
    const int M = Y.rows();
    const int N = Y.cols();

    // Initialise D — N×K random normal columns, then normalise each
    std::mt19937 rng(seed_);
    std::normal_distribution<double> dist(0.0, 1.0);
    D = Eigen::MatrixXd(N, K_);
    for (int k = 0; k < K_; k++) {
        for (int n = 0; n < N; n++) D(n, k) = dist(rng);
        D.col(k).normalize();
    }

    loss.resize(num_iter_);

    OMP omp(N, K_, T0_, Y, batch_size_);

    for (int iter = 0; iter < num_iter_; iter++) {
        // ---- Sparse coding (parallelised inside OMP::do_omp) ----
        omp.D = D;
        omp.do_omp();
        Eigen::MatrixXd X = omp.X;  // M × K

        // ---- Dictionary update (sequential — matches serial K-SVD) ----
        Eigen::MatrixXd E = Y - X * D.transpose();  // M × N

        std::vector<std::vector<int>> atom_signals(K_);
        for (int m = 0; m < M; m++)
            for (int k = 0; k < K_; k++)
                if (X(m, k) != 0.0) atom_signals[k].push_back(m);

        Eigen::MatrixXd E_f(M, N);

        for (int i = 0; i < K_; i++) {
            const auto& filter = atom_signals[i];
            if (filter.empty()) continue;
            const int sz = static_cast<int>(filter.size());

            for (int j = 0; j < sz; j++)
                E.row(filter[j]) += X(filter[j], i) * D.col(i).transpose();

            for (int j = 0; j < sz; j++) E_f.row(j) = E.row(filter[j]);

            Eigen::BDCSVD<Eigen::MatrixXd, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(E_f.topRows(sz));
            D.col(i) = svd.matrixV().col(0);
            for (int j = 0; j < sz; j++)
                X(filter[j], i) = svd.matrixU()(j, 0) * svd.singularValues()(0);

            for (int j = 0; j < sz; j++)
                E.row(filter[j]) -= X(filter[j], i) * D.col(i).transpose();
        }

        loss(iter) = E.norm();
        printf("  iter %d / %d : loss = %.4f\n", iter + 1, num_iter_, loss(iter));
    }
}

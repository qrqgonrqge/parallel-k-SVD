#include "ksvd.hpp"
#include <random>
#include <vector>

KSVD::KSVD(int K, int T0, int num_iter, int batch_size, unsigned int seed)
    : K_(K), T0_(T0), num_iter_(num_iter), batch_size_(batch_size), seed_(seed) {}

void KSVD::fit(const Eigen::MatrixXf& Y) {
    const int M = Y.rows();
    const int N = Y.cols();

    // Initialise D — N×K random normal columns, then normalise each
    std::mt19937 rng(seed_);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    D = Eigen::MatrixXf(N, K_);
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
        Eigen::MatrixXf X = omp.X;  // M × K

        // ---- Dictionary update (sequential — matches serial AK-SVD) ----
        Eigen::MatrixXf E = Y - X * D.transpose();  // M × N

        std::vector<std::vector<int>> atom_signals(K_);
        for (int m = 0; m < M; m++)
            for (int k = 0; k < K_; k++)
                if (X(m, k) != 0.0) atom_signals[k].push_back(m);

        Eigen::MatrixXf E_f(M, N);

        for (int i = 0; i < K_; i++) {
            const auto& filter = atom_signals[i];
            if (filter.empty()) continue;
            const int sz = static_cast<int>(filter.size());

            for (int j = 0; j < sz; j++)
                E.row(filter[j]) += X(filter[j], i) * D.col(i).transpose();

            for (int j = 0; j < sz; j++) E_f.row(j) = E.row(filter[j]);

            Eigen::VectorXf d = (E_f.topRows(sz).transpose() * (E_f.topRows(sz) * D.col(i))).normalized();
            Eigen::VectorXf x_new = E_f.topRows(sz) * d;

            D.col(i) = d;
            for (int j = 0; j < sz; j++)
                X(filter[j], i) = x_new(j);

            for (int j = 0; j < sz; j++)
                E.row(filter[j]) -= X(filter[j], i) * D.col(i).transpose();
        }

        loss(iter) = E.norm();
        // printf("  iter %d / %d : loss = %.4f\n", iter + 1, num_iter_, loss(iter));
    }
}

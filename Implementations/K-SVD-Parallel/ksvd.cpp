#include "ksvd.hpp"
#include <random>
#include <vector>
#include <omp.h>

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
        // ---- Sparse coding ----
        omp.D = D;
        omp.do_omp();
        Eigen::MatrixXd X = omp.X;  // M × K

        // ---- Dictionary update (parallel snapshot) ----
        // Take a single snapshot of the full residual. Each atom independently
        // reads E_base (read-only) and its own column of X and D, so all K_
        // SVDs can run in parallel with no data races.
        // Trade-off vs serial: atoms don't see each other's refined E updates
        // within one iteration, but convergence is near-identical in practice.
        const Eigen::MatrixXd E_base = Y - X * D.transpose();  // M × N, read-only

        // Build non-zero structure
        std::vector<std::vector<int>> atom_signals(K_);
        for (int m = 0; m < M; m++)
            for (int k = 0; k < K_; k++)
                if (X(m, k) != 0.0) atom_signals[k].push_back(m);

        // Per-atom results — each thread writes only to its own index
        std::vector<Eigen::VectorXd> new_d(K_);
        std::vector<Eigen::VectorXd> new_x(K_);

        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < K_; i++) {
            const auto& filter = atom_signals[i];
            if (filter.empty()) continue;
            const int sz = static_cast<int>(filter.size());

            // E_f = E_base rows + atom i's contribution added back (thread-local)
            Eigen::MatrixXd E_f(sz, N);
            for (int j = 0; j < sz; j++)
                E_f.row(j) = E_base.row(filter[j])
                             + X(filter[j], i) * D.col(i).transpose();

            Eigen::BDCSVD<Eigen::MatrixXd, Eigen::ComputeThinU | Eigen::ComputeThinV> svd(E_f);

            new_d[i] = svd.matrixV().col(0);
            new_x[i].resize(sz);
            for (int j = 0; j < sz; j++)
                new_x[i](j) = svd.matrixU()(j, 0) * svd.singularValues()(0);
        }

        // Apply all updates serially (no conflicts — each atom owns its column)
        for (int i = 0; i < K_; i++) {
            if (atom_signals[i].empty()) continue;
            D.col(i) = new_d[i];
            const auto& filter = atom_signals[i];
            for (int j = 0; j < static_cast<int>(filter.size()); j++)
                X(filter[j], i) = new_x[i](j);
        }

        loss(iter) = (Y - X * D.transpose()).norm();
        printf("  iter %d / %d : loss = %.4f\n", iter + 1, num_iter_, loss(iter));
    }
}

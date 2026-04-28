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
        // ---- Sparse coding ----
        omp.D = D;
        omp.do_omp();
        Eigen::MatrixXd X = omp.X;  // M × K

        // ---- Dictionary update ----
        Eigen::MatrixXd E = Y - X * D.transpose();  // M × N

        // Build non-zero structure once: atom_signals[i] = list of signals using atom i.
        // Avoids an O(M×K) scan inside the per-atom loop.
        std::vector<std::vector<int>> atom_signals(K_);
        for (int m = 0; m < M; m++)
            for (int k = 0; k < K_; k++)
                if (X(m, k) != 0.0) atom_signals[k].push_back(m);

        // Pre-allocate E_f at max size — reused each atom, avoids K_ mallocs.
        Eigen::MatrixXd E_f(M, N);

        for (int i = 0; i < K_; i++) {
            const auto& filter = atom_signals[i];
            if (filter.empty()) continue;
            const int sz = static_cast<int>(filter.size());

            // Add atom i's contribution back to the relevant rows of E
            for (int j = 0; j < sz; j++)
                E.row(filter[j]) += X(filter[j], i) * D.col(i).transpose();

            // Gather restricted error matrix into the top sz rows of E_f
            for (int j = 0; j < sz; j++) E_f.row(j) = E.row(filter[j]);

            // One power-iteration step from the current atom (AK-SVD).
            // O(4·sz·N) vs O(sz·N²) for full SVD — ~12× cheaper per atom.
            Eigen::VectorXd d = (E_f.topRows(sz).transpose() * (E_f.topRows(sz) * D.col(i))).normalized();
            Eigen::VectorXd x_new = E_f.topRows(sz) * d;

            D.col(i) = d;
            for (int j = 0; j < sz; j++)
                X(filter[j], i) = x_new(j);

            // Subtract updated contribution back out
            for (int j = 0; j < sz; j++)
                E.row(filter[j]) -= X(filter[j], i) * D.col(i).transpose();
        }

        loss(iter) = E.norm();
        printf("  iter %d / %d : loss = %.4f\n", iter + 1, num_iter_, loss(iter));
    }
}

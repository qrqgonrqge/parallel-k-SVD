#include "omp.hpp"
#include <cstdio>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

OMP::OMP(int N, int K, int T0, Eigen::MatrixXf Y, int batch_size) {
    this->params.N          = N;
    this->params.K          = K;
    this->params.t0         = T0;
    this->params.batch_size = batch_size;
    this->Y                 = Y;
}

bool OMP::do_omp() {
    const int N  = static_cast<int>(this->Y.cols());
    const int M  = static_cast<int>(this->Y.rows());
    const int K  = this->params.K;
    const int T0 = this->params.t0;
    const int bs = this->params.batch_size;

    Eigen::MatrixXf X = Eigen::MatrixXf::Zero(M, K);

    // D is N×K; precompute per-atom norms
    Eigen::VectorXf D_norm(K);
    for (int k = 0; k < K; k++) D_norm(k) = this->D.col(k).norm();

    const int num_batches = (M + bs - 1) / bs;
    tbb::parallel_for(0, num_batches, [&](int b) {
        const int b_start = b * bs;
        const int cur = std::min(bs, M - b_start);

        // y_batch stays fixed; r_batch is the evolving residual
        Eigen::MatrixXf y_batch = this->Y.middleRows(b_start, cur);  // cur × N
        Eigen::MatrixXf r_batch = y_batch;

        // Per-signal Gram-Schmidt state (vector of matrices avoids 3-D indexing)
        std::vector<Eigen::MatrixXf> Q_vec(cur, Eigen::MatrixXf::Zero(N, T0));
        std::vector<Eigen::MatrixXf> R_vec(cur, Eigen::MatrixXf::Zero(T0, T0));
        Eigen::MatrixXf Q_T_y  = Eigen::MatrixXf::Zero(cur, T0);  // cur × T0
        Eigen::MatrixXi I_batch = Eigen::MatrixXi::Zero(cur, T0); // selected atoms

        Eigen::MatrixXf atom_mask = Eigen::MatrixXf::Ones(cur, K);

        // j_stops[i] = first iteration where signal i became linearly dependent
        // (initialised to T0 = ran all the way through)
        Eigen::VectorXi j_stops(cur);
        j_stops.setConstant(T0);
        std::vector<bool> dead(cur, false);

        for (int j = 0; j < T0; j++) {
            // ---- Batched atom selection -----------------------------------------------
            // (cur×N) × (N×K) → (cur×K), masked and abs'd
            Eigen::MatrixXf D_r =
                (r_batch * this->D).cwiseProduct(atom_mask).cwiseAbs();

            // Argmax per row → one selected atom index per signal
            std::vector<int> k_vec(cur);
            for (int i = 0; i < cur; i++) {
                Eigen::Index k_idx;
                D_r.row(i).maxCoeff(&k_idx);
                k_vec[i] = static_cast<int>(k_idx);
            }

            // Gather selected atoms into D_k_batch (cur × N) and update mask/I
            Eigen::MatrixXf D_k_batch(cur, N);
            for (int i = 0; i < cur; i++) {
                atom_mask(i, k_vec[i]) = 0.0;
                I_batch(i, j) = k_vec[i];
                D_k_batch.row(i) = this->D.col(k_vec[i]).transpose();
            }

            // ---- Per-signal Gram-Schmidt -----------------------------------------------
            if (j == 0) {
                for (int i = 0; i < cur; i++) {
                    double norm    = D_norm(k_vec[i]);
                    R_vec[i](0, 0) = norm;
                    Q_vec[i].col(0) = D_k_batch.row(i).transpose() / norm;
                }
            } else {
                for (int i = 0; i < cur; i++) {
                    if (dead[i]) continue;

                    Eigen::VectorXf D_ki = D_k_batch.row(i).transpose();
                    Eigen::VectorXf dot  = Q_vec[i].leftCols(j).transpose() * D_ki;
                    R_vec[i].col(j).head(j) = dot;

                    Eigen::VectorXf q_j      = D_ki - Q_vec[i].leftCols(j) * dot;
                    double          q_j_norm = q_j.norm();

                    if (q_j_norm < 1e-7f) {
                        dead[i]         = true;
                        j_stops[i]      = j;
                        atom_mask(i, k_vec[i]) = 1.0;  // restore so mask stays valid
                        continue;
                    }
                    Q_vec[i].col(j)  = q_j / q_j_norm;
                    R_vec[i](j, j)   = q_j_norm;
                }
            }

            // ---- Batched residual update -----------------------------------------------
            // Build Q_j_batch (cur × N): column j of each signal's Q
            // Dead signals have Q_vec[i].col(j) == 0 (from Zero init), so they
            // contribute zero to the projection and their residual is unchanged.
            Eigen::MatrixXf Q_j_batch(cur, N);
            for (int i = 0; i < cur; i++)
                Q_j_batch.row(i) = Q_vec[i].col(j).transpose();

            // y_proj[i] = y_batch.row(i) · Q_j_batch.row(i)
            Eigen::VectorXf y_proj =
                (y_batch.cwiseProduct(Q_j_batch)).rowwise().sum();  // cur × 1

            Q_T_y.col(j) = y_proj;

            // r_batch.row(i) -= y_proj[i] * Q_j_batch.row(i)  (broadcast over rows)
            r_batch.array() -= Q_j_batch.array().colwise() * y_proj.array();
        }

        // ---- Solve per signal and scatter into X -----------------------------------
        for (int i = 0; i < cur; i++) {
            const int js = j_stops[i];
            if (js == 0) continue;

            Eigen::VectorXf rhs   = Q_T_y.row(i).head(js).transpose();
            Eigen::VectorXf gamma = R_vec[i].topLeftCorner(js, js)
                                            .triangularView<Eigen::Upper>()
                                            .solve(rhs);

            for (int j = 0; j < js; j++)
                X(b_start + i, I_batch(i, j)) = gamma(j);
        }
    });

    this->X = X;
    return true;
}

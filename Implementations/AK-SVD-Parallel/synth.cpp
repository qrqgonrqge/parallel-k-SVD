#include "synth.hpp"
#include <random>
#include <cmath>

Eigen::MatrixXf SynthData::generate(int M, int N, unsigned int seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.f, 1.f);

    Eigen::MatrixXf Y(M, N);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            Y(i, j) = dist(rng);

    for (int j = 0; j < N; j++) {
        float mu = Y.col(j).mean();
        Y.col(j).array() -= mu;
        float s = std::sqrt(Y.col(j).squaredNorm() / M);
        if (s > 1e-7f) Y.col(j) /= s;
    }

    return Y;
}

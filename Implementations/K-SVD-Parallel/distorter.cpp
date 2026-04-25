#include "distorter.hpp"
#include <random>
#include <cmath>

Distorter::Distorter(const Eigen::MatrixXd& image, double noise_level, unsigned int seed)
    : original_(image), distorted_(image) {
    const int H      = image.rows();
    const int W      = image.cols();
    const int half_w = W / 2;

    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    for (int r = 0; r < H; r++)
        for (int c = half_w; c < W; c++)
            distorted_(r, c) += noise_level * dist(rng);
}

Eigen::MatrixXd Distorter::extractNormalizedPatches(int patch_size) const {
    const int H        = distorted_.rows();
    const int half_w   = distorted_.cols() / 2;
    const int n_patches = (H - patch_size + 1) * (half_w - patch_size + 1);
    const int patch_len = patch_size * patch_size;

    Eigen::MatrixXd patches(n_patches, patch_len);

    int idx = 0;
    for (int r = 0; r <= H - patch_size; r++) {
        for (int c = 0; c <= half_w - patch_size; c++) {
            for (int pr = 0; pr < patch_size; pr++)
                for (int pc = 0; pc < patch_size; pc++)
                    patches(idx, pr * patch_size + pc) = distorted_(r + pr, c + pc);
            ++idx;
        }
    }

    // Column-wise zero-mean then unit-std (matches numpy's axis=0 mean/std)
    for (int j = 0; j < patch_len; j++) {
        double mean = patches.col(j).mean();
        patches.col(j).array() -= mean;
        double std = std::sqrt(patches.col(j).squaredNorm() / patches.rows());
        if (std > 1e-10)
            patches.col(j) /= std;
    }

    return patches;
}

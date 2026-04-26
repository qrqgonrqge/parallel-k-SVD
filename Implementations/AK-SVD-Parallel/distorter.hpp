#pragma once
#include <Eigen/Dense>

class Distorter {
public:
    // Adds Gaussian noise (scale noise_level) to the right half of image.
    Distorter(const Eigen::MatrixXf& image, float noise_level = 0.075, unsigned int seed = 42);

    // Extracts all overlapping patch_size×patch_size patches from the left half of the
    // distorted image, flattened row-major, then column-normalised (zero mean, unit std).
    Eigen::MatrixXf extractNormalizedPatches(int patch_size) const;

    const Eigen::MatrixXf& getOriginal()  const { return original_;  }
    const Eigen::MatrixXf& getDistorted() const { return distorted_; }

private:
    Eigen::MatrixXf original_;
    Eigen::MatrixXf distorted_;
};

#include <cstdio>
#include <chrono>
#include <cmath>
#include <vector>
#include <numeric>
#include <Eigen/Dense>
#include "image_process.hpp"
#include "distorter.hpp"
#include "ksvd.hpp"

int main() {
    // ---- Load image ----
    ImageProcess image_process;
    Eigen::MatrixXf image = image_process.loadGrayscaleEigenImage(
        "../images/racoon.jpeg", 256, 192, true);
    printf("Loaded image: %d rows × %d cols\n", (int)image.rows(), (int)image.cols());

    // ---- Distort right half ----
    Distorter distorter(image, 0.075, 42);

    // ---- Extract patches from left half, normalize ----
    const int patch_size = 7;
    Eigen::MatrixXf Y = distorter.extractNormalizedPatches(patch_size);
    printf("Extracted %d patches (%d features)\n\n",
           (int)Y.rows(), (int)Y.cols());

    // ---- Hyperparams ----
    const int K          = 300;
    const int T0         = 20;
    const int num_iter   = 2;
    const int batch_size = 256;
    const int n_runs     = 10;

    printf("K-SVD  K=%d  T0=%d  num_iter=%d  batch_size=%d  n_runs=%d\n\n",
           K, T0, num_iter, batch_size, n_runs);

    // ---- Timing loop ----
    std::vector<double> times(n_runs);
    double final_loss = 0.0;

    for (int run = 0; run < n_runs; run++) {
        KSVD ksvd(K, T0, num_iter, batch_size, /*seed=*/42);

        auto t0 = std::chrono::high_resolution_clock::now();
        ksvd.fit(Y);
        times[run] = std::chrono::duration<double>(
            std::chrono::high_resolution_clock::now() - t0).count();

        final_loss = ksvd.loss(num_iter - 1);
        // printf("  run %2d / %d : %.4f s  (loss %.4f)\n",
        //        run + 1, n_runs, times[run], final_loss);
    }

    // ---- Stats ----
    double mean = std::accumulate(times.begin(), times.end(), 0.0) / n_runs;

    double sq_sum = 0.0;
    for (double t : times) sq_sum += (t - mean) * (t - mean);
    double stddev = std::sqrt(sq_sum / n_runs);

    double tmin = *std::min_element(times.begin(), times.end());
    double tmax = *std::max_element(times.begin(), times.end());

    printf("\n--- Results (%d runs) ---\n", n_runs);
    printf("  mean   : %.4f s\n", mean);
    printf("  std    : %.4f s\n", stddev);
    printf("  min    : %.4f s\n", tmin);
    printf("  max    : %.4f s\n", tmax);
    printf("  loss   : %.4f\n",   final_loss);

    return 0;
}

#include <cstdio>
#include <chrono>
#include <cmath>
#include <vector>
#include <numeric>
#include <string>
#include <memory>
#include <Eigen/Dense>
#include "image_process.hpp"
#include "distorter.hpp"
#include "ksvd.hpp"
#include "synth.hpp"
#include <tbb/global_control.h>
#include <tbb/task_arena.h>
int main(int argc, char* argv[]) {
    int K          = 256;
    int T0         = 32;
    int batch_size = 128;
    int num_iter   = 10;

    // --threads N must be the first argument if present
    std::unique_ptr<tbb::global_control> tbb_ctrl;
    int arg_start = 1;
    if (argc >= 3 && std::string(argv[1]) == "--threads") {
        int n_threads = std::stoi(argv[2]);
        tbb_ctrl = std::make_unique<tbb::global_control>(
            tbb::global_control::max_allowed_parallelism, n_threads);
        arg_start = 3;
    }

    printf("Threads: %zu\n",
           tbb::global_control::active_value(tbb::global_control::max_allowed_parallelism));

    Eigen::MatrixXf Y;

    // --synth M [N]  →  synthetic data
    if (argc >= arg_start + 2 && std::string(argv[arg_start]) == "--synth") {
        int M = std::stoi(argv[arg_start + 1]);
        int N = (argc >= arg_start + 3) ? std::stoi(argv[arg_start + 2]) : 49;
        printf("Synthetic: M=%d  N=%d\n\n", M, N);
        Y = SynthData::generate(M, N);
    } else {
        // ---- Load image ----
        ImageProcess image_process;
        Eigen::MatrixXf image = image_process.loadGrayscaleEigenImage(
            "../images/racoon.jpeg", 256, 192, true);
        printf("Loaded image: %d rows × %d cols\n", (int)image.rows(), (int)image.cols());
        Distorter distorter(image, 0.075, 42);
        const int patch_size = 7;
        Y = distorter.extractNormalizedPatches(patch_size);
        printf("Extracted %d patches (%d features)\n\n",
               (int)Y.rows(), (int)Y.cols());
    }

    // ---- Hyperparams ----
    const int n_runs     = 1;

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

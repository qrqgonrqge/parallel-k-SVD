#pragma once
#include <Eigen/Dense>

class SynthData {
public:
    // Generates M x N matrix: i.i.d. N(0,1), then column zero-mean + unit-std.
    static Eigen::MatrixXf generate(int M, int N, unsigned int seed = 42);
};

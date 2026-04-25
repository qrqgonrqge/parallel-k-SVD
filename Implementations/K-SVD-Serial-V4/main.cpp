#include <Eigen/Dense>
#include "omp.hpp"
#include "image_process.hpp"

int main() {
    // Example parameters - adjust these values as needed
    int N = 100;  // number of atoms
    int K = 64;   // signal dimension
    int T0 = 10;  // sparsity level
    Eigen::MatrixXd Y = Eigen::MatrixXd::Random(K, 100); // example signal matrix
    
    ImageProcess image_process;
    Eigen::MatrixXd image = image_process.loadGrayscaleEigenImage("../images/racoon.jpeg",
                                    256, 256, true);

    //print image data:
    std::cout << image << std::endl;
    
    // save image (TEST)
    // image_process.saveEigenGrayImage(image, "output.png");

    OMP omp(N, K, T0, Y);
    omp.do_omp();
    return 0;
}
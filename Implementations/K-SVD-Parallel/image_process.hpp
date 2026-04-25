#include <Eigen/Dense>
#include <string>

class ImageProcess {
public:
    Eigen::MatrixXd loadGrayscaleEigenImage(
        const std::string& image_path,
        int target_width = -1,
        int target_height = -1,
        bool normalize = true
    );
    
    void saveEigenGrayImage(const Eigen::MatrixXd& img, const std::string& filename);
};
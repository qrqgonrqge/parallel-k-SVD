#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <stdexcept>
#include <string>
#include "image_process.hpp"

Eigen::MatrixXd ImageProcess::loadGrayscaleEigenImage(
    const std::string& image_path,
    int target_width,
    int target_height,
    bool normalize
) {
    // Load image unchanged first
    cv::Mat img = cv::imread(image_path, cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }

    // Convert to grayscale if needed
    cv::Mat gray;
    if (img.channels() == 1) {
        gray = img;
    } else if (img.channels() == 3) {
        cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else if (img.channels() == 4) {
        cv::cvtColor(img, gray, cv::COLOR_BGRA2GRAY);
    } else {
        throw std::runtime_error("Unsupported number of channels in image: " + image_path);
    }

    // Resize if requested
    if (target_width > 0 && target_height > 0) {
        cv::resize(gray, gray, cv::Size(target_width, target_height), 0, 0, cv::INTER_AREA);
    } else if ((target_width > 0) != (target_height > 0)) {
        throw std::runtime_error("Must provide both target_width and target_height, or neither.");
    }

    // Convert to double
    cv::Mat gray64;
    if (normalize) {
        gray.convertTo(gray64, CV_64F, 1.0 / 255.0);
    } else {
        gray.convertTo(gray64, CV_64F);
    }

    // Copy into Eigen matrix
    Eigen::MatrixXd out(gray64.rows, gray64.cols);
    for (int r = 0; r < gray64.rows; ++r) {
        for (int c = 0; c < gray64.cols; ++c) {
            out(r, c) = gray64.at<double>(r, c);
        }
    }

    return out;
}

void ImageProcess::saveEigenGrayImage(const Eigen::MatrixXd& img, const std::string& filename) {
    if (img.size() == 0) {
        throw std::runtime_error("Empty image");
    }

    // Convert Eigen → cv::Mat (double)
    cv::Mat tmp(img.rows(), img.cols(), CV_64F);
    for (int r = 0; r < img.rows(); ++r) {
        for (int c = 0; c < img.cols(); ++c) {
            tmp.at<double>(r, c) = img(r, c);
        }
    }

    // Scale [0,1] → [0,255]
    cv::Mat out;
    tmp.convertTo(out, CV_8UC1, 255.0);

    if (!cv::imwrite(filename, out)) {
        throw std::runtime_error("Failed to save image");
    }
}
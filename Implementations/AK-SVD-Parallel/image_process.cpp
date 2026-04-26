#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <stdexcept>
#include <string>
#include "image_process.hpp"

Eigen::MatrixXf ImageProcess::loadGrayscaleEigenImage(
    const std::string& image_path,
    int target_width,
    int target_height,
    bool normalize
) {
    cv::Mat img = cv::imread(image_path, cv::IMREAD_UNCHANGED);
    if (img.empty()) {
        throw std::runtime_error("Failed to load image: " + image_path);
    }

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

    if (target_width > 0 && target_height > 0) {
        cv::resize(gray, gray, cv::Size(target_width, target_height), 0, 0, cv::INTER_AREA);
    } else if ((target_width > 0) != (target_height > 0)) {
        throw std::runtime_error("Must provide both target_width and target_height, or neither.");
    }

    cv::Mat gray32;
    if (normalize) {
        gray.convertTo(gray32, CV_32F, 1.0f / 255.0f);
    } else {
        gray.convertTo(gray32, CV_32F);
    }

    Eigen::MatrixXf out(gray32.rows, gray32.cols);
    for (int r = 0; r < gray32.rows; ++r)
        for (int c = 0; c < gray32.cols; ++c)
            out(r, c) = gray32.at<float>(r, c);

    return out;
}

void ImageProcess::saveEigenGrayImage(const Eigen::MatrixXf& img, const std::string& filename) {
    if (img.size() == 0) {
        throw std::runtime_error("Empty image");
    }

    cv::Mat tmp(img.rows(), img.cols(), CV_32F);
    for (int r = 0; r < img.rows(); ++r)
        for (int c = 0; c < img.cols(); ++c)
            tmp.at<float>(r, c) = img(r, c);

    cv::Mat out;
    tmp.convertTo(out, CV_8UC1, 255.0f);

    if (!cv::imwrite(filename, out)) {
        throw std::runtime_error("Failed to save image");
    }
}

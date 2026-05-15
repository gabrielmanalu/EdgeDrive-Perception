/**
 * heatmap_generator.cpp
 * C++ equivalent of Ultralytics Solutions Heatmap
 */

#include "heatmap_generator.hpp"
#include <algorithm>

HeatmapGenerator::HeatmapGenerator(int width, int height,
                                    float decay, int sigma)
    : w_(width), h_(height), decay_(decay), sigma_(sigma)
{
    accumulator_ = cv::Mat::zeros(h_, w_, CV_32F);
    buildKernel();
}

void HeatmapGenerator::buildKernel() {
    int ks = sigma_ * 4 + 1;  // kernel size (odd)
    kernel_ = cv::Mat(ks, ks, CV_32F);
    float center = ks / 2.0f;
    float sigma2 = 2.0f * sigma_ * sigma_;
    float sum = 0.0f;

    for (int y = 0; y < ks; y++)
        for (int x = 0; x < ks; x++) {
            float dx = x - center, dy = y - center;
            float v = std::exp(-(dx*dx + dy*dy) / sigma2);
            kernel_.at<float>(y, x) = v;
            sum += v;
        }
    kernel_ /= sum;  // normalize
}

void HeatmapGenerator::update(const std::vector<Detection>& dets) {
    // Per-frame decay — old detections fade out
    if (decay_ < 1.0f)
        accumulator_ *= decay_;

    int ks   = kernel_.rows;
    int half = ks / 2;

    for (const auto& d : dets) {
        int cx = static_cast<int>((d.x1 + d.x2) * 0.5f);
        int cy = static_cast<int>((d.y1 + d.y2) * 0.5f);

        // ROI bounds clamped to accumulator
        int x0 = cx - half, y0 = cy - half;
        int x1 = x0 + ks,   y1 = y0 + ks;

        // Kernel sub-region if near edge
        int kx0 = std::max(0, -x0), ky0 = std::max(0, -y0);
        x0 = std::max(0, x0); y0 = std::max(0, y0);
        x1 = std::min(w_, x1); y1 = std::min(h_, y1);

        if (x1 <= x0 || y1 <= y0) continue;

        cv::Rect roi(x0, y0, x1-x0, y1-y0);
        cv::Mat k_roi = kernel_(
            cv::Rect(kx0, ky0, x1-x0, y1-y0));
        accumulator_(roi) += k_roi;

        total_count_++;
    }
}

cv::Mat HeatmapGenerator::render(const cv::Mat& frame, float alpha) const {
    // Normalize accumulator to [0, 255]
    double min_v, max_v;
    cv::minMaxLoc(accumulator_, &min_v, &max_v);

    cv::Mat normalized;
    if (max_v > 0)
        accumulator_.convertTo(normalized, CV_8U,
                               255.0 / max_v);
    else
        normalized = cv::Mat::zeros(h_, w_, CV_8U);

    // Apply COLORMAP_JET: blue=cold → red=hot
    cv::Mat colormap;
    cv::applyColorMap(normalized, colormap, cv::COLORMAP_JET);

    // Alpha blend with original frame
    cv::Mat result;
    cv::addWeighted(colormap, alpha, frame, 1.0f - alpha, 0, result);

    // Counter overlay
    std::string txt = "Detections: " + std::to_string(total_count_);
    cv::putText(result, txt,
                cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(255, 255, 255), 2, cv::LINE_AA);

    return result;
}

void HeatmapGenerator::reset() {
    accumulator_.setTo(0);
    total_count_ = 0;
}
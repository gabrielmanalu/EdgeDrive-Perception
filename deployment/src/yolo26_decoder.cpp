/**
 * yolo26_decoder.cpp — YOLO26n Output Decoder
 * =============================================
 * Decodes raw TensorRT output buffer into Detection structs.
 *
 * YOLO26n output format:
 *   Tensor shape: [1, 300, 6]
 *   Per detection: [x1, y1, x2, y2, confidence, class_id]
 *   Coordinates: normalized to input tensor size (640x640)
 *
 * Key difference from YOLOv8n:
 *   YOLO26n uses an NMS-free detection head (anchor-free + no NMS).
 *   Output is already 300 deduplicated proposals — no NMS needed here.
 *   This is why YOLO26n has lower post-processing latency on Jetson.
 *
 * Python reference:
 *   ultralytics/models/yolo/detect/predict.py → postprocess()
 *   Output shape confirmed: (1, 300, 6) from our TensorRT export
 */

#include "yolo26_decoder.hpp"
#include <algorithm>

// ── Class names and colors ────────────────────────────────────────────────────
// Must match calibration.yaml names order exactly

const std::vector<std::string> YOLO26Decoder::CLASS_NAMES = {
    "car",          // 0
    "pedestrian",   // 1
    "bicycle",      // 2
    "motorcycle",   // 3
    "bus",          // 4
    "truck",        // 5
    "traffic_cone", // 6
    "barrier"       // 7
};

const std::vector<cv::Scalar> YOLO26Decoder::CLASS_COLORS = {
    cv::Scalar(255, 255,   0),  // car          — cyan
    cv::Scalar(  0,   0, 255),  // pedestrian   — red
    cv::Scalar(  0, 255,   0),  // bicycle      — green
    cv::Scalar(  0, 255,   0),  // motorcycle   — green
    cv::Scalar(  0, 255, 255),  // bus          — yellow
    cv::Scalar(  0, 165, 255),  // truck        — orange
    cv::Scalar(255, 255, 255),  // traffic_cone — white
    cv::Scalar(255,   0, 255),  // barrier      — magenta
};

// ── Constructor ───────────────────────────────────────────────────────────────

YOLO26Decoder::YOLO26Decoder(float score_thresh)
    : score_thresh_(score_thresh) {}

// ── Decode ────────────────────────────────────────────────────────────────────

std::vector<Detection> YOLO26Decoder::decode(
    const float* output_data,
    int orig_w, int orig_h)
{
    /**
     * Output tensor layout: [1, 300, 6]
     * Flattened: output_data[i * 6 + j] where i=proposal, j=field
     *
     * Fields:
     *   j=0: x1 (in 640x640 input space)
     *   j=1: y1
     *   j=2: x2
     *   j=3: y2
     *   j=4: confidence score
     *   j=5: class_id (float, cast to int)
     */

    constexpr int NUM_PROPOSALS = 300;
    constexpr int FIELDS_PER_DET = 6;

    // Scale factors: 640x640 → original image size
    const float scale_x = static_cast<float>(orig_w) / 640.0f;
    const float scale_y = static_cast<float>(orig_h) / 640.0f;

    std::vector<Detection> detections;
    detections.reserve(32); // pre-allocate for typical detection count

    for (int i = 0; i < NUM_PROPOSALS; i++) {
        const float* det = output_data + i * FIELDS_PER_DET;

        float score    = det[4];
        int   class_id = static_cast<int>(det[5]);

        // Filter by confidence threshold
        if (score < score_thresh_) continue;

        // Filter invalid class ids
        if (class_id < 0 ||
            class_id >= static_cast<int>(CLASS_NAMES.size())) continue;

        // Scale coordinates to original image space
        float x1 = det[0] * scale_x;
        float y1 = det[1] * scale_y;
        float x2 = det[2] * scale_x;
        float y2 = det[3] * scale_y;

        // Clip to image bounds
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(orig_w)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(orig_h)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(orig_w)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(orig_h)));

        // Skip degenerate boxes
        if (x2 <= x1 || y2 <= y1) continue;

        Detection d;
        d.x1         = x1;
        d.y1         = y1;
        d.x2         = x2;
        d.y2         = y2;
        d.score      = score;
        d.class_id   = class_id;
        d.class_name = CLASS_NAMES[class_id];

        detections.push_back(d);
    }

    return detections;
}

// ── Draw ──────────────────────────────────────────────────────────────────────

cv::Mat YOLO26Decoder::draw(const cv::Mat& frame,
                             const std::vector<Detection>& dets)
{
    cv::Mat out = frame.clone();

    for (const auto& d : dets) {
        cv::Scalar color = (d.class_id < static_cast<int>(CLASS_COLORS.size()))
                         ? CLASS_COLORS[d.class_id]
                         : cv::Scalar(255, 255, 255);

        // Bounding box
        cv::rectangle(out,
                      cv::Point(static_cast<int>(d.x1),
                                static_cast<int>(d.y1)),
                      cv::Point(static_cast<int>(d.x2),
                                static_cast<int>(d.y2)),
                      color, 2);

        // Label: "car 0.94"
        std::string label = d.class_name + " " +
                            std::to_string(static_cast<int>(d.score * 100)) + "%";

        int baseline = 0;
        cv::Size text_size = cv::getTextSize(
            label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

        // Label background
        cv::rectangle(out,
                      cv::Point(static_cast<int>(d.x1),
                                static_cast<int>(d.y1) - text_size.height - 4),
                      cv::Point(static_cast<int>(d.x1) + text_size.width,
                                static_cast<int>(d.y1)),
                      color, cv::FILLED);

        // Label text
        cv::putText(out, label,
                    cv::Point(static_cast<int>(d.x1),
                              static_cast<int>(d.y1) - 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }

    return out;
}
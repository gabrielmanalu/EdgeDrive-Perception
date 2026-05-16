#pragma once

#include <vector>
#include <string>
#include <opencv2/opencv.hpp>

// ── Detection struct ──────────────────────────────────────────────────────────

struct Detection {
    float x1, y1, x2, y2;  // bounding box in original image coordinates
    float score;            // confidence score
    int   class_id;         // class index
    std::string class_name; // class label
};

// ── YOLO26n decoder ───────────────────────────────────────────────────────────

class YOLO26Decoder {
public:
    /**
     * Decodes YOLO26n TensorRT output into Detection structs.
     * Auto-detects output format from output_size:
     *
     * Format A [1, 300, 6]   — end2end=True (post-NMS)
     *   output_size = 1800
     *   Fields: [x1, y1, x2, y2, score, class_id]
     *
     * Format B [1, 12, 8400] — end2end=False (raw head, INT8 default)
     *   output_size = 100800
     *   Layout: [batch, 4+num_classes, num_anchors]
     *   Requires sigmoid on class scores + NMS in decoder
     */
    YOLO26Decoder(float score_thresh = 0.3f, float nms_thresh = 0.3f);

    // Decode raw TRT output into detections
    // output_size: total float count (used to detect format)
    // orig_w, orig_h: original frame dimensions for coordinate scaling
    std::vector<Detection> decode(
        const float* output_data,
        int output_size,
        int orig_w, int orig_h
    );

    // Draw detections on image (returns annotated copy)
    cv::Mat draw(const cv::Mat& frame,
                 const std::vector<Detection>& dets);

    // Debug: print channel statistics to understand output tensor layout
    void debugOutput(const float* data, int output_size) const;

private:
    float score_thresh_;
    float nms_thresh_;

    // Format A decoder — post-NMS [1, 300, 6]
    std::vector<Detection> decodeEndToEnd(
        const float* data, int orig_w, int orig_h);

    // Format B decoder — raw head [1, 12, 8400]
    std::vector<Detection> decodeRaw(
        const float* data, int orig_w, int orig_h);

    // Greedy class-agnostic NMS
    std::vector<Detection> nms(std::vector<Detection>& dets) const;

public:
    // nuScenes class names and colors — public for BEVVisualizer
    static const std::vector<std::string> CLASS_NAMES;
    static const std::vector<cv::Scalar>  CLASS_COLORS;
};
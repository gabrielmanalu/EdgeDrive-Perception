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
     * Decode YOLO26n TensorRT output into Detection structs.
     *
     * YOLO26n output tensor shape: [1, 300, 6]
     *   - 300 proposals (NMS-free head, fixed number)
     *   - 6 values per proposal: [x1, y1, x2, y2, score, class_id]
     *
     * Coordinates are in input tensor space (640x640).
     * Must scale back to original image dimensions.
     *
     * Note: YOLO26n uses NMS-free detection head (unlike YOLOv8n).
     * No NMS post-processing needed — output already deduplicated.
     */
    YOLO26Decoder(float score_thresh = 0.3f);

    // Decode raw output buffer into detections
    // output_data : pointer to TRT output buffer [1 * 300 * 6 floats]
    // orig_w, orig_h : original image dimensions for coordinate scaling
    std::vector<Detection> decode(
        const float* output_data,
        int orig_w, int orig_h
    );

    // Draw detections on image (returns annotated copy)
    cv::Mat draw(const cv::Mat& frame,
                 const std::vector<Detection>& dets);

private:
    float score_thresh_;

    // nuScenes class names (must match calibration.yaml order)
    static const std::vector<std::string> CLASS_NAMES;

    // Colors per class for visualization
    static const std::vector<cv::Scalar> CLASS_COLORS;
};
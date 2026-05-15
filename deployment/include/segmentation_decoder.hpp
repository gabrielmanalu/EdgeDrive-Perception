#pragma once
#include "yolo26_decoder.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * SegmentationDecoder — YOLO26n-seg output decoder
 * ==================================================
 * Decodes the two-output segmentation model:
 *
 * Output 0 — Detection head [1, 44, 8400]
 *   channels 0-3  : cx, cy, w, h  (cxcywh, 640×640 space)
 *   channels 4-11 : class probabilities (sigmoid applied)
 *   channels 12-43: mask coefficients (32 floats per anchor)
 *
 * Output 1 — Prototype masks [1, 32, 160, 160]
 *   32 prototype masks at 1/4 input resolution
 *
 * Mask generation per detection:
 *   mask = sigmoid(coefficients × prototypes)  [160×160]
 *   mask = resize(mask, orig_size)              [orig_h×orig_w]
 *   mask = mask > 0.5                          [binary]
 *   mask = mask & bbox_region                  [clip to box]
 *
 * Requires YOLO26n-seg engine (weights/yolo26n_seg_int8.engine).
 *
 * Note: segmentation adds ~1-2ms decode overhead vs detection-only.
 * At 193 FPS inference, segmentation decode runs in <2ms.
 */

struct SegResult {
    Detection   det;        // bounding box + class + score
    cv::Mat     mask;       // CV_8U binary mask (orig resolution)
    cv::Scalar  color;      // class color for visualization
};

class SegmentationDecoder {
public:
    static constexpr int NUM_CLASSES    = 8;
    static constexpr int NUM_MASK_COEF  = 32;
    static constexpr int NUM_ANCHORS    = 8400;
    static constexpr int PROTO_H        = 160;
    static constexpr int PROTO_W        = 160;

    // Channels per anchor: 4 box + 8 class + 32 mask = 44
    static constexpr int CHANNELS = 4 + NUM_CLASSES + NUM_MASK_COEF;

    SegmentationDecoder(float score_thresh = 0.3f,
                        float nms_thresh   = 0.3f);

    /**
     * Decode segmentation outputs.
     * @param output0  Detection head [1, 44, 8400] — total: 369600 floats
     * @param output1  Prototype masks [1, 32, 160, 160] — total: 819200 floats
     * @param orig_w   Original frame width
     * @param orig_h   Original frame height
     */
    std::vector<SegResult> decode(
        const float* output0,
        const float* output1,
        int orig_w, int orig_h);

    // Draw segmentation masks + boxes on frame
    cv::Mat draw(const cv::Mat& frame,
                 const std::vector<SegResult>& results,
                 float mask_alpha = 0.4f) const;

private:
    float score_thresh_;
    float nms_thresh_;

    // Generate binary mask for one detection
    cv::Mat generateMask(
        const float* coefs,        // 32 coefficients
        const float* protos,       // [32, 160, 160]
        const Detection& det,
        int orig_w, int orig_h) const;

    // Letterbox inverse for mask coordinates
    float gain_;
    float pad_x_, pad_y_;
    void updateLetterbox(int orig_w, int orig_h);
};
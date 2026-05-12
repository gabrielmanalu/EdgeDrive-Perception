/**
 * yolo26_decoder.cpp — YOLO26n Output Decoder
 * =============================================
 * Decodes raw TensorRT output into Detection structs.
 *
 * Handles TWO output formats depending on export settings:
 *
 * Format A — End-to-end [1, 300, 6] (end2end=True):
 *   Post-NMS output. Each row: [x1, y1, x2, y2, score, class_id]
 *   Only available on FP16 or when JetPack 6 INT8 bug is fixed.
 *
 * Format B — Raw head [1, 12, 8400] (end2end=False, default INT8):
 *   Raw YOLO detection head output. No NMS applied.
 *   Layout: [batch, 4+num_classes, num_anchors]
 *     - Channels 0-3  : box [x1, y1, x2, y2] in input (640×640) space
 *     - Channels 4-11 : class logits (apply sigmoid for probability)
 *   8400 anchors = 80×80 + 40×40 + 20×20 (P3 + P4 + P5 feature maps)
 *   Requires: threshold filtering + NMS in decoder (this file).
 *
 * Format B is used by default because:
 *   TensorRT 10.3.0 on JetPack 6 disables end2end branch for INT8:
 *   "WARNING ⚠️ TensorRT 10.3.0 on JetPack 6 with int8 has known
 *    end2end build issues, disabling end2end branch."
 *
 * Auto-detection: output_size == 300*6=1800 → Format A
 *                 output_size == 12*8400=100800 → Format B
 */

#include "yolo26_decoder.hpp"
#include <algorithm>
#include <numeric>

// ── Constants ─────────────────────────────────────────────────────────────────

static constexpr int NUM_ANCHORS    = 8400;   // 80x80 + 40x40 + 20x20
static constexpr int NUM_CLASSES    = 8;
static constexpr int RAW_CHANNELS   = 4 + NUM_CLASSES;  // 12
static constexpr int END2END_DETS   = 300;
static constexpr int END2END_FIELDS = 6;      // [x1,y1,x2,y2,score,class_id]

// ── Class names and colors ────────────────────────────────────────────────────
// Must match calibration.yaml order exactly

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

YOLO26Decoder::YOLO26Decoder(float score_thresh, float nms_thresh)
    : score_thresh_(score_thresh), nms_thresh_(nms_thresh) {}

// ── Decode ────────────────────────────────────────────────────────────────────

std::vector<Detection> YOLO26Decoder::decode(
    const float* output_data,
    int output_size,
    int orig_w, int orig_h)
{
    // Auto-detect output format from total element count
    if (output_size == END2END_DETS * END2END_FIELDS) {
        return decodeEndToEnd(output_data, orig_w, orig_h);
    } else {
        return decodeRaw(output_data, orig_w, orig_h);
    }
}

// ── Format A decoder — [1, 300, 6] ───────────────────────────────────────────

std::vector<Detection> YOLO26Decoder::decodeEndToEnd(
    const float* data, int orig_w, int orig_h)
{
    const float scale_x = static_cast<float>(orig_w) / 640.0f;
    const float scale_y = static_cast<float>(orig_h) / 640.0f;

    std::vector<Detection> dets;
    dets.reserve(32);

    for (int i = 0; i < END2END_DETS; i++) {
        const float* d = data + i * END2END_FIELDS;
        float score    = d[4];
        int   class_id = static_cast<int>(d[5]);

        if (score < score_thresh_) continue;
        if (class_id < 0 || class_id >= NUM_CLASSES) continue;

        Detection det;
        det.x1         = std::max(0.0f, std::min(d[0] * scale_x, (float)orig_w));
        det.y1         = std::max(0.0f, std::min(d[1] * scale_y, (float)orig_h));
        det.x2         = std::max(0.0f, std::min(d[2] * scale_x, (float)orig_w));
        det.y2         = std::max(0.0f, std::min(d[3] * scale_y, (float)orig_h));
        det.score      = score;
        det.class_id   = class_id;
        det.class_name = CLASS_NAMES[class_id];

        if (det.x2 > det.x1 && det.y2 > det.y1)
            dets.push_back(det);
    }
    return dets;
}

// ── Debug helper ──────────────────────────────────────────────────────────────

void YOLO26Decoder::debugOutput(const float* data, int output_size) const {
    printf("\n=== Output Tensor Debug ===\n");
    printf("Total floats: %d\n", output_size);

    if (output_size == NUM_ANCHORS * RAW_CHANNELS) {
        printf("Format: [1, 12, 8400] raw head\n\n");
        // Print min/max for each of the 12 channels
        for (int c = 0; c < RAW_CHANNELS; c++) {
            float min_v = 1e9f, max_v = -1e9f, sum = 0.0f;
            for (int a = 0; a < NUM_ANCHORS; a++) {
                float v = data[c * NUM_ANCHORS + a];
                min_v = std::min(min_v, v);
                max_v = std::max(max_v, v);
                sum  += v;
            }
            printf("  ch[%2d] min=%8.4f  max=%8.4f  mean=%8.4f  %s\n",
                   c, min_v, max_v, sum / NUM_ANCHORS,
                   c < 4 ? "(box coord)" : "(class score)");
        }
        // Print first anchor's full row
        printf("\nAnchor[0] raw values:\n");
        for (int c = 0; c < RAW_CHANNELS; c++) {
            printf("  [%2d] = %f\n", c, data[c * NUM_ANCHORS + 0]);
        }
    } else {
        printf("Format: [1, 300, 6] end2end\n");
        printf("First detection: %.3f %.3f %.3f %.3f  score=%.3f  class=%d\n",
               data[0], data[1], data[2], data[3], data[4], (int)data[5]);
    }
    printf("===========================\n\n");
}

// ── Format B decoder — [1, 12, 8400] ─────────────────────────────────────────

std::vector<Detection> YOLO26Decoder::decodeRaw(
    const float* data, int orig_w, int orig_h)
{
    /**
     * Layout: data[channel * NUM_ANCHORS + anchor]
     *
     * Channels 0-3  : x1, y1, x2, y2 in 640×640 input space
     * Channels 4-11 : class logits → sigmoid → probability
     *
     * Steps:
     *   1. For each anchor: find max class score after sigmoid
     *   2. Filter by score threshold
     *   3. Scale boxes to original image size
     *   4. Apply NMS per class
     */

    const float scale_x = static_cast<float>(orig_w) / 640.0f;
    const float scale_y = static_cast<float>(orig_h) / 640.0f;

    std::vector<Detection> candidates;
    candidates.reserve(256);

    for (int a = 0; a < NUM_ANCHORS; a++) {
        // Box format is [cx, cy, w, h] in 640×640 input space
        // NOT [x1, y1, x2, y2] as previously assumed.
        // Confirmed by debug: ch[0] mean=319 (cx ≈ image center),
        //                     ch[2] mean=75  (w, small object width)
        float cx = data[0 * NUM_ANCHORS + a];
        float cy = data[1 * NUM_ANCHORS + a];
        float w  = data[2 * NUM_ANCHORS + a];
        float h  = data[3 * NUM_ANCHORS + a];

        // Convert cxcywh → xyxy
        float x1 = cx - w * 0.5f;
        float y1 = cy - h * 0.5f;
        float x2 = cx + w * 0.5f;
        float y2 = cy + h * 0.5f;

        // Skip degenerate boxes (w or h <= 0)
        if (w <= 0.0f || h <= 0.0f) continue;

        // Class scores are already probabilities [0,1] — NO sigmoid needed.
        // The model applies sigmoid internally in the detection head.
        // Confirmed by debug: ch[4-11] values in range [0.0, 0.86]
        // Applying sigmoid on top would give sigmoid(0.0001)=0.500025
        // → every anchor "passes" threshold 0.3 → hundreds of false boxes.
        float max_score = 0.0f;
        int   class_id  = 0;
        for (int c = 0; c < NUM_CLASSES; c++) {
            float score = data[(4 + c) * NUM_ANCHORS + a];  // already [0,1]
            if (score > max_score) {
                max_score = score;
                class_id  = c;
            }
        }

        if (max_score < score_thresh_) continue;

        // Scale to original image
        Detection det;
        det.x1 = std::max(0.0f, std::min(x1 * scale_x, (float)orig_w));
        det.y1 = std::max(0.0f, std::min(y1 * scale_y, (float)orig_h));
        det.x2 = std::max(0.0f, std::min(x2 * scale_x, (float)orig_w));
        det.y2 = std::max(0.0f, std::min(y2 * scale_y, (float)orig_h));
        det.score      = max_score;
        det.class_id   = class_id;
        det.class_name = CLASS_NAMES[class_id];

        candidates.push_back(det);
    }

    // Apply NMS
    return nms(candidates);
}

// ── NMS ───────────────────────────────────────────────────────────────────────

static float iou(const Detection& a, const Detection& b) {
    float ix1 = std::max(a.x1, b.x1);
    float iy1 = std::max(a.y1, b.y1);
    float ix2 = std::min(a.x2, b.x2);
    float iy2 = std::min(a.y2, b.y2);

    float inter = std::max(0.0f, ix2 - ix1) * std::max(0.0f, iy2 - iy1);
    if (inter == 0.0f) return 0.0f;

    float area_a = (a.x2 - a.x1) * (a.y2 - a.y1);
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (area_a + area_b - inter);
}

std::vector<Detection> YOLO26Decoder::nms(
    std::vector<Detection>& dets) const
{
    if (dets.empty()) return {};

    // Sort by score descending
    std::sort(dets.begin(), dets.end(),
              [](const Detection& a, const Detection& b) {
                  return a.score > b.score;
              });

    std::vector<bool> suppressed(dets.size(), false);
    std::vector<Detection> result;
    result.reserve(dets.size());

    for (size_t i = 0; i < dets.size(); i++) {
        if (suppressed[i]) continue;
        result.push_back(dets[i]);

        for (size_t j = i + 1; j < dets.size(); j++) {
            if (suppressed[j]) continue;
            // Class-agnostic NMS — suppress any class if boxes overlap heavily
            if (iou(dets[i], dets[j]) > nms_thresh_) {
                suppressed[j] = true;
            }
        }
    }

    return result;
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

        cv::rectangle(out,
                      cv::Point(static_cast<int>(d.x1),
                                static_cast<int>(d.y1)),
                      cv::Point(static_cast<int>(d.x2),
                                static_cast<int>(d.y2)),
                      color, 2);

        std::string label = d.class_name + " " +
                            std::to_string(static_cast<int>(d.score * 100)) + "%";

        int baseline = 0;
        cv::Size ts = cv::getTextSize(
            label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);

        cv::rectangle(out,
                      cv::Point(static_cast<int>(d.x1),
                                static_cast<int>(d.y1) - ts.height - 4),
                      cv::Point(static_cast<int>(d.x1) + ts.width,
                                static_cast<int>(d.y1)),
                      color, cv::FILLED);

        cv::putText(out, label,
                    cv::Point(static_cast<int>(d.x1),
                              static_cast<int>(d.y1) - 2),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 0, 0), 1, cv::LINE_AA);
    }

    return out;
}
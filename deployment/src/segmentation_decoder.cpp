/**
 * segmentation_decoder.cpp
 * C++ equivalent of Ultralytics YOLO26n-seg postprocessing
 */

#include "segmentation_decoder.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>

static const std::vector<cv::Scalar> SEG_COLORS = {
    cv::Scalar(255, 255,   0),  // car
    cv::Scalar(  0,   0, 255),  // pedestrian
    cv::Scalar(  0, 255,   0),  // bicycle
    cv::Scalar(  0, 255,   0),  // motorcycle
    cv::Scalar(  0, 255, 255),  // bus
    cv::Scalar(  0, 165, 255),  // truck
    cv::Scalar(255, 255, 255),  // traffic_cone
    cv::Scalar(255,   0, 255),  // barrier
};

SegmentationDecoder::SegmentationDecoder(float score_thresh, float nms_thresh)
    : score_thresh_(score_thresh), nms_thresh_(nms_thresh),
      gain_(1.0f), pad_x_(0.0f), pad_y_(0.0f) {}

void SegmentationDecoder::updateLetterbox(int orig_w, int orig_h) {
    gain_  = std::min(640.0f / orig_w, 640.0f / orig_h);
    pad_x_ = (640.0f - orig_w * gain_) / 2.0f;
    pad_y_ = (640.0f - orig_h * gain_) / 2.0f;
}

// ── Mask generation ───────────────────────────────────────────────────────────

cv::Mat SegmentationDecoder::generateMask(
    const float* coefs,
    const float* protos,
    const Detection& det,
    int orig_w, int orig_h) const
{
    // mask = sigmoid(coefs @ protos)  [160×160]
    cv::Mat mask_160(PROTO_H, PROTO_W, CV_32F, 0.0f);

    for (int k = 0; k < NUM_MASK_COEF; k++) {
        // protos layout: [32, 160, 160]
        const float* proto_k = protos + k * PROTO_H * PROTO_W;
        float coef_k = coefs[k];
        for (int i = 0; i < PROTO_H * PROTO_W; i++)
            mask_160.at<float>(i / PROTO_W, i % PROTO_W) +=
                coef_k * proto_k[i];
    }

    // Sigmoid activation
    for (int i = 0; i < PROTO_H; i++)
        for (int j = 0; j < PROTO_W; j++) {
            float v = mask_160.at<float>(i, j);
            mask_160.at<float>(i, j) = 1.0f / (1.0f + std::exp(-v));
        }

    // Scale bounding box to proto space (160×160)
    float scale = static_cast<float>(PROTO_H) / 640.0f;
    int mx1 = std::max(0, static_cast<int>((det.x1 * gain_ + pad_x_) * scale));
    int my1 = std::max(0, static_cast<int>((det.y1 * gain_ + pad_y_) * scale));
    int mx2 = std::min(PROTO_W, static_cast<int>((det.x2 * gain_ + pad_x_) * scale));
    int my2 = std::min(PROTO_H, static_cast<int>((det.y2 * gain_ + pad_y_) * scale));

    // Crop mask to bounding box region
    cv::Mat cropped = cv::Mat::zeros(PROTO_H, PROTO_W, CV_32F);
    if (mx2 > mx1 && my2 > my1)
        mask_160(cv::Rect(mx1, my1, mx2-mx1, my2-my1))
            .copyTo(cropped(cv::Rect(mx1, my1, mx2-mx1, my2-my1)));

    // Resize to original image size
    cv::Mat mask_full;
    cv::resize(cropped, mask_full, cv::Size(orig_w, orig_h),
               0, 0, cv::INTER_LINEAR);

    // Threshold → binary mask
    cv::Mat binary;
    cv::threshold(mask_full, binary, 0.5f, 255.0f, cv::THRESH_BINARY);
    binary.convertTo(binary, CV_8U);

    return binary;
}

// ── Decode ────────────────────────────────────────────────────────────────────

std::vector<SegResult> SegmentationDecoder::decode(
    const float* output0,
    const float* output1,
    int orig_w, int orig_h)
{
    updateLetterbox(orig_w, orig_h);

    // ── Step 1: filter candidates from output0 [1, 44, 8400] ──────────────

    struct Candidate {
        float x1, y1, x2, y2;
        float score;
        int   class_id;
        float coefs[NUM_MASK_COEF];
    };
    std::vector<Candidate> candidates;

    for (int a = 0; a < NUM_ANCHORS; a++) {
        float cx = output0[0 * NUM_ANCHORS + a];
        float cy = output0[1 * NUM_ANCHORS + a];
        float w  = output0[2 * NUM_ANCHORS + a];
        float h  = output0[3 * NUM_ANCHORS + a];
        if (w <= 0 || h <= 0) continue;

        // Class scores
        float max_s = 0; int cls = 0;
        for (int c = 0; c < NUM_CLASSES; c++) {
            float s = output0[(4 + c) * NUM_ANCHORS + a];
            if (s > max_s) { max_s = s; cls = c; }
        }
        if (max_s < score_thresh_) continue;

        // Letterbox inverse → original coords
        float x1 = (cx - w*0.5f - pad_x_) / gain_;
        float y1 = (cy - h*0.5f - pad_y_) / gain_;
        float x2 = (cx + w*0.5f - pad_x_) / gain_;
        float y2 = (cy + h*0.5f - pad_y_) / gain_;
        x1 = std::max(0.0f, std::min(x1, (float)orig_w));
        y1 = std::max(0.0f, std::min(y1, (float)orig_h));
        x2 = std::max(0.0f, std::min(x2, (float)orig_w));
        y2 = std::max(0.0f, std::min(y2, (float)orig_h));
        if (x2 <= x1 || y2 <= y1) continue;

        Candidate c_;
        c_.x1 = x1; c_.y1 = y1; c_.x2 = x2; c_.y2 = y2;
        c_.score = max_s; c_.class_id = cls;
        for (int k = 0; k < NUM_MASK_COEF; k++)
            c_.coefs[k] = output0[(4 + NUM_CLASSES + k) * NUM_ANCHORS + a];
        candidates.push_back(c_);
    }

    // ── Step 2: class-agnostic NMS ─────────────────────────────────────────

    std::sort(candidates.begin(), candidates.end(),
              [](auto& a, auto& b){ return a.score > b.score; });

    std::vector<bool> suppressed(candidates.size(), false);
    std::vector<int>  kept;

    for (size_t i = 0; i < candidates.size(); i++) {
        if (suppressed[i]) continue;
        kept.push_back(i);
        auto& a = candidates[i];
        float area_a = (a.x2-a.x1)*(a.y2-a.y1);
        for (size_t j = i+1; j < candidates.size(); j++) {
            if (suppressed[j]) continue;
            auto& b = candidates[j];
            float ix1 = std::max(a.x1, b.x1), iy1 = std::max(a.y1, b.y1);
            float ix2 = std::min(a.x2, b.x2), iy2 = std::min(a.y2, b.y2);
            float inter = std::max(0.0f, ix2-ix1)*std::max(0.0f, iy2-iy1);
            float area_b = (b.x2-b.x1)*(b.y2-b.y1);
            if (inter / (area_a + area_b - inter) > nms_thresh_)
                suppressed[j] = true;
        }
    }

    // ── Step 3: generate masks for surviving detections ────────────────────

    std::vector<SegResult> results;
    for (int idx : kept) {
        auto& cand = candidates[idx];
        Detection det;
        det.x1 = cand.x1; det.y1 = cand.y1;
        det.x2 = cand.x2; det.y2 = cand.y2;
        det.score    = cand.score;
        det.class_id = cand.class_id;

        SegResult r;
        r.det   = det;
        r.color = (cand.class_id < (int)SEG_COLORS.size())
                ? SEG_COLORS[cand.class_id]
                : cv::Scalar(255,255,255);
        r.mask  = generateMask(cand.coefs, output1, det, orig_w, orig_h);
        results.push_back(std::move(r));
    }

    return results;
}

// ── Draw ──────────────────────────────────────────────────────────────────────

cv::Mat SegmentationDecoder::draw(const cv::Mat& frame,
                                   const std::vector<SegResult>& results,
                                   float mask_alpha) const
{
    cv::Mat out = frame.clone();

    for (const auto& r : results) {
        // Colored mask overlay
        if (!r.mask.empty()) {
            cv::Mat colored(frame.rows, frame.cols, CV_8UC3, r.color);
            cv::Mat overlay = out.clone();
            colored.copyTo(overlay, r.mask);
            cv::addWeighted(overlay, mask_alpha, out, 1.0f-mask_alpha, 0, out);
        }

        // Bounding box
        cv::rectangle(out,
                      cv::Point(r.det.x1, r.det.y1),
                      cv::Point(r.det.x2, r.det.y2),
                      r.color, 1);

        // Label
        std::string lbl = r.det.class_name + " " +
            std::to_string(static_cast<int>(r.det.score*100)) + "%";
        cv::putText(out, lbl,
                    cv::Point(r.det.x1, r.det.y1 - 3),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45,
                    r.color, 1, cv::LINE_AA);
    }

    return out;
}
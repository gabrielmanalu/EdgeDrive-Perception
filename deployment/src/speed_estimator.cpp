/**
 * speed_estimator.cpp
 * C++ equivalent of Ultralytics Solutions SpeedEstimator
 */

#include "speed_estimator.hpp"
#include <cmath>
#include <numeric>
#include <sstream>
#include <iomanip>

SpeedEstimator::SpeedEstimator(float px_per_m, float fps, int history)
    : px_per_m_(px_per_m), fps_(fps), history_(history) {}

// ── Track matching ────────────────────────────────────────────────────────────

int SpeedEstimator::matchOrCreate(cv::Point2f centroid, int class_id,
                                   float box_w, float box_h)
{
    // Match by distance — threshold scales with box size
    float thresh = std::max(box_w, box_h) * 0.6f;
    float best_d = thresh;
    int   best_id = -1;

    for (auto& [id, t] : tracks_) {
        if (t.class_id != class_id) continue;
        float d = cv::norm(centroid - t.centroid);
        if (d < best_d) { best_d = d; best_id = id; }
    }

    if (best_id >= 0) {
        auto& t = tracks_[best_id];
        t.centroid = centroid;
        t.miss     = 0;
        t.history.push_back(centroid);
        if ((int)t.history.size() > history_)
            t.history.pop_front();
        computeSpeed(t);
        return best_id;
    }

    int id = next_id_++;
    Track t;
    t.centroid = centroid;
    t.class_id = class_id;
    t.history.push_back(centroid);
    tracks_[id] = t;
    return id;
}

// ── Speed computation ─────────────────────────────────────────────────────────

void SpeedEstimator::computeSpeed(Track& t) {
    if (t.history.size() < 2) { t.speed_kmh = 0.0f; return; }

    // Average displacement over history window
    float total_px = 0.0f;
    int   n        = static_cast<int>(t.history.size());
    for (int i = 1; i < n; i++) {
        float dx = t.history[i].x - t.history[i-1].x;
        float dy = t.history[i].y - t.history[i-1].y;
        total_px += std::sqrt(dx*dx + dy*dy);
    }
    float px_per_frame = total_px / (n - 1);

    // px/frame → m/s → km/h
    float m_per_frame  = px_per_frame / px_per_m_;
    float m_per_second = m_per_frame  * fps_;
    t.speed_kmh = m_per_second * 3.6f;
}

// ── Update ────────────────────────────────────────────────────────────────────

void SpeedEstimator::update(const std::vector<Detection>& dets) {
    for (auto& [id, t] : tracks_) t.miss++;

    // Remove stale
    for (auto it = tracks_.begin(); it != tracks_.end(); ) {
        if (it->second.miss > 5) it = tracks_.erase(it);
        else ++it;
    }

    for (const auto& d : dets) {
        cv::Point2f c((d.x1+d.x2)*0.5f, (d.y1+d.y2)*0.5f);
        matchOrCreate(c, d.class_id, d.x2-d.x1, d.y2-d.y1);
    }
}

// ── Draw ──────────────────────────────────────────────────────────────────────

cv::Mat SpeedEstimator::draw(const cv::Mat& frame) const {
    cv::Mat out = frame.clone();

    for (const auto& [id, t] : tracks_) {
        if (t.speed_kmh <= 0.0f) continue;

        std::ostringstream ss;
        ss << std::fixed << std::setprecision(0)
           << t.speed_kmh << " km/h";
        std::string label = ss.str();

        cv::Point pt(static_cast<int>(t.centroid.x),
                     static_cast<int>(t.centroid.y) - 12);

        int base;
        cv::Size ts = cv::getTextSize(
            label, cv::FONT_HERSHEY_SIMPLEX, 0.55, 1, &base);

        cv::rectangle(out,
                      cv::Point(pt.x - 2, pt.y - ts.height - 3),
                      cv::Point(pt.x + ts.width + 2, pt.y + 3),
                      cv::Scalar(0, 0, 0), cv::FILLED);

        // Color by speed: green < 30 < yellow < 60 < red
        cv::Scalar col = t.speed_kmh < 30
            ? cv::Scalar(80,  220, 80)
            : t.speed_kmh < 60
            ? cv::Scalar(80,  220, 220)
            : cv::Scalar(80,  80,  220);

        cv::putText(out, label, pt,
                    cv::FONT_HERSHEY_SIMPLEX, 0.55,
                    col, 1, cv::LINE_AA);

        // Trail
        int n = static_cast<int>(t.history.size());
        for (int i = 1; i < n; i++) {
            float alpha = static_cast<float>(i) / n;
            cv::line(out,
                     cv::Point(static_cast<int>(t.history[i-1].x),
                               static_cast<int>(t.history[i-1].y)),
                     cv::Point(static_cast<int>(t.history[i].x),
                               static_cast<int>(t.history[i].y)),
                     cv::Scalar(0, 200*alpha, 255*alpha),
                     1, cv::LINE_AA);
        }
    }

    return out;
}

float SpeedEstimator::speedKmh(int track_id) const {
    auto it = tracks_.find(track_id);
    return it != tracks_.end() ? it->second.speed_kmh : -1.0f;
}
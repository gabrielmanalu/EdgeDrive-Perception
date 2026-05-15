/**
 * object_counter.cpp
 * C++ equivalent of Ultralytics Solutions ObjectCounter
 */

#include "object_counter.hpp"
#include <cmath>
#include <sstream>

// ── Line / zone setup ─────────────────────────────────────────────────────────

void ObjectCounter::setLine(cv::Point p1, cv::Point p2) {
    line_p1_ = p1;
    line_p2_ = p2;
    has_line_ = true;
}

void ObjectCounter::setZone(const std::vector<cv::Point>& polygon) {
    zone_ = polygon;
    has_zone_ = true;
}

// ── Line side helper ──────────────────────────────────────────────────────────

float ObjectCounter::lineSide(cv::Point2f pt) const {
    // Cross product of (p2-p1) × (pt-p1)
    float dx = line_p2_.x - line_p1_.x;
    float dy = line_p2_.y - line_p1_.y;
    return dx * (pt.y - line_p1_.y) - dy * (pt.x - line_p1_.x);
}

// ── Simple centroid tracker ───────────────────────────────────────────────────

int ObjectCounter::matchOrCreate(cv::Point2f centroid, int class_id) {
    float best_dist = 80.0f;  // max pixel distance to match
    int   best_id   = -1;

    for (auto& [id, t] : tracks_) {
        float d = cv::norm(centroid - t.centroid);
        if (d < best_dist) {
            best_dist = d;
            best_id   = id;
        }
    }

    if (best_id >= 0) {
        tracks_[best_id].centroid = centroid;
        tracks_[best_id].class_id = class_id;
        tracks_[best_id].miss     = 0;
        return best_id;
    }

    // New track
    int id = next_id_++;
    tracks_[id] = {centroid, class_id, 0};
    return id;
}

// ── Update ────────────────────────────────────────────────────────────────────

void ObjectCounter::update(const std::vector<Detection>& dets) {
    // Age existing tracks
    for (auto& [id, t] : tracks_) t.miss++;

    // Remove stale tracks (missed > 5 frames)
    for (auto it = tracks_.begin(); it != tracks_.end(); ) {
        if (it->second.miss > 5) {
            last_side_.erase(it->first);
            it = tracks_.erase(it);
        } else ++it;
    }

    // Zone count resets each frame
    count_zone_ = 0;

    for (const auto& d : dets) {
        cv::Point2f c((d.x1 + d.x2) * 0.5f, (d.y1 + d.y2) * 0.5f);
        int id = matchOrCreate(c, d.class_id);

        // Line crossing detection
        if (has_line_) {
            float side = lineSide(c);
            auto it = last_side_.find(id);
            if (it != last_side_.end()) {
                // Crossed if sign changed
                if (it->second * side < 0) {
                    if (side > 0) count_in_++;
                    else          count_out_++;
                }
            }
            last_side_[id] = side;
        }

        // Zone counting
        if (has_zone_) {
            if (cv::pointPolygonTest(zone_, c, false) >= 0)
                count_zone_++;
        }
    }
}

// ── Draw ──────────────────────────────────────────────────────────────────────

cv::Mat ObjectCounter::draw(const cv::Mat& frame) const {
    cv::Mat out = frame.clone();

    // Draw counting line
    if (has_line_) {
        cv::line(out, line_p1_, line_p2_,
                 cv::Scalar(0, 255, 255), 2, cv::LINE_AA);

        // IN/OUT labels at line midpoint
        cv::Point mid((line_p1_.x + line_p2_.x) / 2,
                      (line_p1_.y + line_p2_.y) / 2);

        auto label = [&](std::string txt, int dy, cv::Scalar col) {
            int base;
            cv::Size ts = cv::getTextSize(txt,
                cv::FONT_HERSHEY_SIMPLEX, 0.65, 2, &base);
            cv::rectangle(out,
                          cv::Point(mid.x - 4, mid.y + dy - ts.height - 4),
                          cv::Point(mid.x + ts.width + 4, mid.y + dy + 4),
                          cv::Scalar(0, 0, 0), cv::FILLED);
            cv::putText(out, txt,
                        cv::Point(mid.x, mid.y + dy),
                        cv::FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv::LINE_AA);
        };

        label("IN:  " + std::to_string(count_in_),  -10,
              cv::Scalar(80, 220, 80));
        label("OUT: " + std::to_string(count_out_),  20,
              cv::Scalar(80, 80, 220));
    }

    // Draw zone
    if (has_zone_ && !zone_.empty()) {
        cv::polylines(out, zone_, true,
                      cv::Scalar(255, 165, 0), 2, cv::LINE_AA);
        cv::putText(out,
                    "Zone: " + std::to_string(count_zone_),
                    zone_[0] + cv::Point(5, -5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.65,
                    cv::Scalar(255, 165, 0), 2, cv::LINE_AA);
    }

    // Track centroids
    for (const auto& [id, t] : tracks_) {
        cv::circle(out,
                   cv::Point(static_cast<int>(t.centroid.x),
                             static_cast<int>(t.centroid.y)),
                   3, cv::Scalar(0, 255, 255), cv::FILLED);
    }

    return out;
}

void ObjectCounter::reset() {
    count_in_ = count_out_ = count_zone_ = 0;
    tracks_.clear();
    last_side_.clear();
    next_id_ = 0;
}
#pragma once
#include "yolo26_decoder.hpp"
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>
#include <string>

/**
 * ObjectCounter — Line and zone crossing counter
 * ================================================
 * C++ equivalent of Ultralytics Solutions ObjectCounter.
 *
 * Line crossing:
 *   Counts objects that cross a defined line segment.
 *   Tracks direction (IN / OUT) based on which side the
 *   centroid was on in the previous frame.
 *
 * Zone counting:
 *   Counts objects whose centroid is inside a polygon.
 *
 * Usage (line):
 *   ObjectCounter counter;
 *   counter.setLine(cv::Point(0, 360), cv::Point(1280, 360));
 *   while (running) {
 *       auto dets = decoder.decode(...);
 *       counter.update(dets);
 *       cv::Mat vis = counter.draw(frame);
 *   }
 */
class ObjectCounter {
public:
    ObjectCounter() = default;

    // Define counting line (two endpoints)
    void setLine(cv::Point p1, cv::Point p2);

    // Define counting zone (polygon vertices)
    void setZone(const std::vector<cv::Point>& polygon);

    // Update with current frame detections
    // Uses simple centroid matching (IoU-based) for track continuity
    void update(const std::vector<Detection>& dets);

    // Draw line/zone + counts on frame
    cv::Mat draw(const cv::Mat& frame) const;

    // Reset counts
    void reset();

    int countIn()  const { return count_in_; }
    int countOut() const { return count_out_; }
    int countZone() const { return count_zone_; }

private:
    // Line mode
    cv::Point line_p1_, line_p2_;
    bool      has_line_ = false;

    // Zone mode
    std::vector<cv::Point> zone_;
    bool                   has_zone_ = false;

    // Counts
    int count_in_   = 0;
    int count_out_  = 0;
    int count_zone_ = 0;

    // Track state: track_id → last centroid
    int next_id_ = 0;
    struct Track {
        cv::Point2f centroid;
        int         class_id;
        int         miss = 0;
    };
    std::unordered_map<int, Track> tracks_;

    // Simple centroid matching
    int matchOrCreate(cv::Point2f centroid, int class_id);

    // Returns +1 if point is on positive side of line, -1 otherwise
    float lineSide(cv::Point2f pt) const;
    std::unordered_map<int, float> last_side_;
};
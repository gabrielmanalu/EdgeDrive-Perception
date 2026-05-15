#pragma once
#include "yolo26_decoder.hpp"
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <deque>
#include <vector>

/**
 * SpeedEstimator — Object speed estimation from monocular camera
 * ==============================================================
 * C++ equivalent of Ultralytics Solutions SpeedEstimator.
 *
 * Method:
 *   1. Track objects frame-to-frame via centroid IoU matching
 *   2. Measure pixel displacement per frame
 *   3. Convert to km/h using pixel-per-meter calibration factor
 *
 * Limitation (same as Python version):
 *   Assumes stationary camera. On a moving vehicle, measured speed
 *   is relative to the camera (object speed - ego speed).
 *   For absolute speed, subtract ego velocity from odometry/IMU.
 *   See docs/solutions_on_edge.md for full analysis.
 *
 * Calibration:
 *   px_per_m = known_object_width_pixels / known_object_width_meters
 *   e.g. a car (~1.8m wide) spanning 90px at 10m → px_per_m = 50
 *
 * Usage:
 *   SpeedEstimator estimator(50.0f, 30.0f);  // px/m, fps
 *   while (running) {
 *       auto dets = decoder.decode(...);
 *       estimator.update(dets);
 *       cv::Mat vis = estimator.draw(annotated);
 *   }
 */
class SpeedEstimator {
public:
    SpeedEstimator(float px_per_m = 50.0f,  // pixels per meter at ref distance
                   float fps      = 30.0f,  // camera frame rate
                   int   history  = 10);    // frames to smooth speed over

    // Update with current detections
    void update(const std::vector<Detection>& dets);

    // Draw speed labels on frame
    cv::Mat draw(const cv::Mat& frame) const;

    // Get speed for a track id (km/h), -1 if unknown
    float speedKmh(int track_id) const;

private:
    float px_per_m_;
    float fps_;
    int   history_;
    int   next_id_ = 0;

    struct Track {
        cv::Point2f centroid;
        int         class_id;
        int         miss = 0;
        std::deque<cv::Point2f> history;  // recent centroids
        float speed_kmh = 0.0f;
    };
    std::unordered_map<int, Track> tracks_;

    int matchOrCreate(cv::Point2f centroid, int class_id,
                      float box_w, float box_h);
    void computeSpeed(Track& t);
};
#pragma once

#include "yolo26_decoder.hpp"
#include <opencv2/opencv.hpp>
#include <vector>

/**
 * BEVVisualizer — Bird's Eye View from monocular camera detections
 * =================================================================
 *
 * Projects 2D bounding box bottom-centers to ground plane using the
 * pinhole camera model with a flat-ground assumption.
 *
 * Projection math (per detection):
 *   bottom-center pixel: u = (x1+x2)/2,  v = y2
 *   depth (forward):     Z = h * fy / (v - cy)      [meters]
 *   lateral offset:      X = (u - cx) * Z / fx       [meters]
 *
 *   where h  = camera height above ground (default: 1.5m)
 *         fx,fy = focal lengths (pixels)
 *         cx,cy = principal point (pixels)
 *
 * Limitations:
 *   - Requires flat road surface (z=0 assumption)
 *   - Accuracy degrades for objects far from ground plane
 *     (elevated trucks, buses on slopes)
 *   - Distances are estimates without proper extrinsic calibration
 *   - Camera pitch assumed ~0° (horizontal mounting)
 *
 * Camera matrix K:
 *   Default estimated for 1280×720 USB webcam (~70° diagonal FOV):
 *     fx = fy = 910,  cx = 640,  cy = 360
 *   Pass actual calibrated K for metric accuracy.
 */
class BEVVisualizer {
public:
    /**
     * @param K     3×3 camera intrinsic matrix (CV_64F)
     *              Pass cv::Mat() to use default estimated values.
     * @param cam_h Camera height above ground in meters (default: 1.2m dashcam)
     */
    explicit BEVVisualizer(const cv::Mat& K    = cv::Mat(),
                           float          cam_h = 1.2f);

    /**
     * Render BEV map for current frame detections.
     *
     * @param dets      Detections from YOLO26Decoder
     * @param img_w     Original frame width (pixels)
     * @param img_h     Original frame height (pixels)
     * @return          BEV canvas (bev_h_ × bev_w_, BGR)
     */
    cv::Mat render(const std::vector<Detection>& dets,
                   int img_w, int img_h);

    /**
     * Combine camera view and BEV side by side.
     * Both panels scaled to same height.
     */
    cv::Mat sideBySide(const cv::Mat& camera_frame,
                       const cv::Mat& bev_frame) const;

    // BEV canvas dimensions (pixels)
    static constexpr int BEV_W = 480;
    static constexpr int BEV_H = 620;

    // World space range
    static constexpr float RANGE_FORWARD_M = 15.0f; // meters ahead
    static constexpr float RANGE_LATERAL_M = 10.0f; // meters each side

private:
    cv::Mat K_;
    float   cam_h_;
    float   fx_, fy_, cx_, cy_;

    // BEV scale: pixels per meter
    float   scale_x_;  // BEV_W / (2 * RANGE_LATERAL_M)
    float   scale_y_;  // BEV_H / RANGE_FORWARD_M

    // Ego position on BEV canvas (bottom center)
    int     ego_x_;
    int     ego_y_;

    // Project image point (u,v) → BEV canvas (px,py)
    // Returns false if point is behind camera or out of range
    bool project(float u, float v,
                 int& bev_x, int& bev_y,
                 float& dist_m) const;

    // Draw static background (grid, ego vehicle, lane markers)
    cv::Mat makeBackground() const;

    // Draw one detection on BEV canvas
    void drawDetection(cv::Mat& canvas,
                       const Detection& det,
                       float u, float v) const;
};
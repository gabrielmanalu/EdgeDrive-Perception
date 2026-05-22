/*
 * fusion_node.hpp — EdgeDrive Perception: Camera-LiDAR Late Fusion Node
 * =======================================================================
 *
 * Fuses camera 2D detections with LiDAR 3D detections in Bird's Eye View.
 *
 * Subscribes (ApproximateTime synchronized):
 *   /detections/camera  (vision_msgs/Detection2DArray)
 *   /detections/lidar   (vision_msgs/Detection3DArray)
 *
 * Publishes:
 *   /detections/fused        (vision_msgs/Detection3DArray)
 *   /visualization/fused     (visualization_msgs/MarkerArray)
 *     red    = camera + LiDAR matched (fused)
 *     green  = camera only (no LiDAR match)
 *     blue   = LiDAR only  (no camera match)
 *
 * Parameters:
 *   match_threshold   : max BEV distance for association (default: 3.0m)
 *   camera_height     : camera height above ground (default: 1.2m)
 *   camera_fx/fy      : focal lengths (default: 640.0)
 *   camera_cx/cy      : principal point (default: 0 = auto)
 *   sync_tolerance    : ApproximateTime tolerance seconds (default: 0.3)
 *
 * Algorithm:
 *   1. Camera detections → BEV (x,y) via pinhole ground plane projection
 *   2. LiDAR detections  → BEV (x,y) already in base_link frame
 *   3. Hungarian algorithm on distance cost matrix
 *   4. Matched pairs → fused detection (LiDAR 3D + camera class/score)
 *   5. Unmatched → kept as single-sensor detections
 */

#pragma once

#include <rclcpp/rclcpp.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <memory>
#include <string>
#include <vector>

namespace edgedrive {

// BEV point for matching
struct BEVPoint {
    float x, y;           // position in base_link frame
    int   orig_idx;       // index in original detection array
    float score;
    std::string class_id;
};

class FusionNode : public rclcpp::Node {
public:
    explicit FusionNode(
        const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
    ~FusionNode() = default;

private:
    // Parameters
    float  match_threshold_;
    float  camera_height_;
    float  fx_, fy_, cx_, cy_;

    // Message filter types
    using CamMsg   = vision_msgs::msg::Detection2DArray;
    using LidarMsg = vision_msgs::msg::Detection3DArray;
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<
                           CamMsg, LidarMsg>;
    using Synchronizer = message_filters::Synchronizer<SyncPolicy>;

    message_filters::Subscriber<CamMsg>   cam_sub_;
    message_filters::Subscriber<LidarMsg> lidar_sub_;
    std::shared_ptr<Synchronizer>         sync_;

    // Publishers
    rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr   fused_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

    // Synchronized callback
    void fusionCallback(
        const CamMsg::ConstSharedPtr&   cam_msg,
        const LidarMsg::ConstSharedPtr& lidar_msg);

    // Project camera 2D detection to BEV (x,y) via ground plane
    std::vector<BEVPoint> cameraDetsToBEV(
        const CamMsg& msg, int img_w, int img_h);

    // Extract BEV (x,y) from LiDAR 3D detections
    std::vector<BEVPoint> lidarDetsToBEV(const LidarMsg& msg);

    // Hungarian algorithm — returns assignment vector
    // assignment[i] = j means camera[i] matched to lidar[j], -1 = unmatched
    std::vector<int> hungarianMatch(
        const std::vector<BEVPoint>& cam_bev,
        const std::vector<BEVPoint>& lidar_bev);

    // Publish fused MarkerArray
    void publishMarkers(
        const std::vector<BEVPoint>& cam_bev,
        const std::vector<BEVPoint>& lidar_bev,
        const std::vector<int>&      assignment,
        const std_msgs::msg::Header& header);

    int frame_count_ = 0;
};

} // namespace edgedrive
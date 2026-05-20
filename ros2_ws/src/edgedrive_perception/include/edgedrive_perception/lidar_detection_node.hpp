/*
 * lidar_detection_node.hpp — EdgeDrive Perception: LiDAR Detection ROS2 Node
 * ============================================================================
 *
 * Wraps CUDA-PointPillars pipeline in a ROS2 node.
 *
 * Subscribes:
 *   /nuscenes/lidar/pointcloud  (sensor_msgs/PointCloud2)
 *
 * Publishes:
 *   /detections/lidar          (vision_msgs/Detection3DArray)
 *   /detections/lidar_markers  (visualization_msgs/MarkerArray) → RViz2 blue cylinders
 *
 * Parameters:
 *   engine_path    : TRT plan file (default: cuda-pointpillars/model/pointpillar.plan)
 *   score_threshold: detection confidence threshold (default: 0.15)
 *   publish_markers: publish RViz2 MarkerArray (default: true)
 *
 * nuScenes voxelization params (hardcoded to match our exported ONNX):
 *   range    : [-50,-50,-5, 50,50,3] m
 *   voxel    : [0.25, 0.25, 8.0] m
 *   grid     : 400 × 400
 *   features : 4 (x, y, z, intensity — ring channel stripped from PointCloud2)
 */

#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <vision_msgs/msg/detection3_d_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <cuda_runtime.h>
#include <memory>
#include <string>
#include <vector>

// Forward declare to avoid pulling all CUDA-PP headers into the header
namespace pointpillar { namespace lidar {
    class Core;
    struct BoundingBox;
}}

namespace edgedrive {

class LidarDetectionNode : public rclcpp::Node {
public:
    explicit LidarDetectionNode(
        const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
    ~LidarDetectionNode();

private:
    // Parameters
    std::string engine_path_;
    float       score_threshold_;
    bool        publish_markers_;

    // CUDA-PointPillars core
    std::shared_ptr<pointpillar::lidar::Core> pp_core_;
    cudaStream_t stream_ = nullptr;

    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr lidar_sub_;

    // Publishers
    rclcpp::Publisher<vision_msgs::msg::Detection3DArray>::SharedPtr    det_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr  marker_pub_;

    // Callbacks
    void lidarCallback(const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg);

    // Publish MarkerArray (blue cylinders in RViz2 BEV)
    void publishMarkers(
        const std::vector<pointpillar::lidar::BoundingBox>& boxes,
        const std_msgs::msg::Header& header);

    // Stats
    int frame_count_ = 0;
};

} // namespace edgedrive
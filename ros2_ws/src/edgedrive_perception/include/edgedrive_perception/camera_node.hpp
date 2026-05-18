#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <cv_bridge/cv_bridge.h>

#include "trt_engine.hpp"
#include "yolo26_decoder.hpp"
#include "bev_visualizer.hpp"
#include "profiler.hpp"

#include <memory>
#include <string>

namespace edgedrive {

/**
 * CameraNode — YOLO26n TRT INT8 detection ROS2 node
 * ==================================================
 *
 * Subscribes:
 *   /camera/image_raw  (sensor_msgs/Image)
 *
 * Publishes:
 *   /detections/camera         (vision_msgs/Detection2DArray)
 *   /camera/annotated          (sensor_msgs/Image)  — optional viz
 *   /camera/bev                (sensor_msgs/Image)  — optional BEV image
 *   /detections/camera_markers (visualization_msgs/MarkerArray) — RViz2 BEV
 *
 * Parameters:
 *   engine_path    : path to TRT engine file
 *   score_threshold: detection confidence threshold (default: 0.3)
 *   publish_viz    : publish annotated image (default: true)
 *   publish_bev    : publish BEV image (default: false)
 *   publish_markers: publish RViz2 MarkerArray (default: true)
 *   camera_height  : camera height above ground in meters (default: 1.2)
 *   camera_fx      : focal length x in pixels (default: 640 = 90° HFOV)
 *   camera_fy      : focal length y in pixels (default: 640)
 *   camera_cx      : principal point x (default: image_width/2)
 *   camera_cy      : principal point y (default: image_height/2)
 */
class CameraNode : public rclcpp::Node {
public:
    explicit CameraNode(const rclcpp::NodeOptions& options = rclcpp::NodeOptions());
    ~CameraNode() = default;

private:
    // Parameters
    std::string engine_path_;
    float       score_threshold_;
    bool        publish_viz_;
    bool        publish_bev_;
    bool        publish_markers_;
    float       camera_height_;
    float       fx_, fy_, cx_, cy_;

    // Inference modules
    std::unique_ptr<TRTEngine>      engine_;
    std::unique_ptr<YOLO26Decoder>  decoder_;
    std::unique_ptr<BEVVisualizer>  bev_;
    std::unique_ptr<Profiler>       profiler_;

    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;

    // Publishers
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr       det_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr                  viz_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr                  bev_pub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr     marker_pub_;

    // Callbacks
    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);

    // Convert detections to ROS2 message
    vision_msgs::msg::Detection2DArray toRosMsg(
        const std::vector<Detection>& dets,
        const std_msgs::msg::Header& header);

    // Project detections to ground plane and publish as MarkerArray
    // Uses same pinhole model as BEVVisualizer:
    //   Z = cam_h * fy / (v - cy)   [forward distance, meters]
    //   X = (u - cx) * Z / fx       [lateral offset, meters]
    void publishMarkers(
        const std::vector<Detection>& dets,
        const std_msgs::msg::Header& header,
        int img_w, int img_h);

    // Stats logging
    int frame_count_ = 0;
    int marker_id_   = 0;
};

} // namespace edgedrive
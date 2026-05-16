#pragma once

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>
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
 *   /detections/camera     (vision_msgs/Detection2DArray)
 *   /camera/annotated      (sensor_msgs/Image)  — optional viz
 *   /camera/bev            (sensor_msgs/Image)  — optional BEV
 *
 * Parameters:
 *   engine_path    : path to TRT engine file
 *   score_threshold: detection confidence threshold (default: 0.3)
 *   publish_viz    : publish annotated image (default: true)
 *   publish_bev    : publish BEV image (default: false)
 *   camera_height  : camera height above ground in meters (default: 1.2)
 *
 * Usage:
 *   ros2 run edgedrive_perception camera_node \
 *       --ros-args \
 *       -p engine_path:=/workspace/weights/yolo26n_det_int8_raw.engine \
 *       -p score_threshold:=0.3 \
 *       -p publish_bev:=true
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
    float       camera_height_;

    // Inference modules
    std::unique_ptr<TRTEngine>      engine_;
    std::unique_ptr<YOLO26Decoder>  decoder_;
    std::unique_ptr<BEVVisualizer>  bev_;
    std::unique_ptr<Profiler>       profiler_;

    // Subscribers
    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;

    // Publishers
    rclcpp::Publisher<vision_msgs::msg::Detection2DArray>::SharedPtr det_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr             viz_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr             bev_pub_;

    // Callback
    void imageCallback(const sensor_msgs::msg::Image::ConstSharedPtr& msg);

    // Convert detections to ROS2 message
    vision_msgs::msg::Detection2DArray toRosMsg(
        const std::vector<Detection>& dets,
        const std_msgs::msg::Header& header);

    // Stats logging
    int frame_count_ = 0;
};

} // namespace edgedrive
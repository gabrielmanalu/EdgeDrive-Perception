/**
 * camera_node.cpp — YOLO26n TRT INT8 ROS2 Detection Node
 * ========================================================
 *
 * Subscribes to /camera/image_raw, runs TensorRT inference,
 * publishes Detection2DArray + optional annotated image + BEV.
 */

#include "edgedrive_perception/camera_node.hpp"

#include <vision_msgs/msg/detection2_d.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <cv_bridge/cv_bridge.h>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace edgedrive {

// ── Constructor ───────────────────────────────────────────────────────────────

CameraNode::CameraNode(const rclcpp::NodeOptions& options)
    : Node("camera_node", options)
{
    // ── Declare parameters ────────────────────────────────────────────────────
    engine_path_      = declare_parameter("engine_path",
        "/workspace/weights/yolo26n_det_int8_raw.engine");
    score_threshold_  = declare_parameter("score_threshold", 0.3f);
    publish_viz_      = declare_parameter("publish_viz", true);
    publish_bev_      = declare_parameter("publish_bev", false);
    publish_markers_  = declare_parameter("publish_markers", true);
    camera_height_    = declare_parameter("camera_height", 1.2f);
    fx_               = declare_parameter("camera_fx", 640.0f);
    fy_               = declare_parameter("camera_fy", 640.0f);
    cx_               = declare_parameter("camera_cx", 0.0f);  // 0 = auto
    cy_               = declare_parameter("camera_cy", 0.0f);  // 0 = auto

    RCLCPP_INFO(get_logger(), "EdgeDrive CameraNode starting...");
    RCLCPP_INFO(get_logger(), "  Engine    : %s", engine_path_.c_str());
    RCLCPP_INFO(get_logger(), "  Threshold : %.2f", score_threshold_);
    RCLCPP_INFO(get_logger(), "  Publish viz    : %s", publish_viz_ ? "yes" : "no");
    RCLCPP_INFO(get_logger(), "  Publish BEV    : %s", publish_bev_ ? "yes" : "no");
    RCLCPP_INFO(get_logger(), "  Publish markers: %s", publish_markers_ ? "yes" : "no");

    // ── Initialize inference modules ──────────────────────────────────────────
    try {
        engine_  = std::make_unique<TRTEngine>(engine_path_);
        decoder_ = std::make_unique<YOLO26Decoder>(score_threshold_);
        profiler_= std::make_unique<Profiler>(30);

        if (publish_bev_)
            bev_ = std::make_unique<BEVVisualizer>(cv::Mat(), camera_height_);

        RCLCPP_INFO(get_logger(), "TRT engine loaded successfully.");
    } catch (const std::exception& e) {
        RCLCPP_FATAL(get_logger(), "Failed to load engine: %s", e.what());
        throw;
    }

    // ── Publishers ────────────────────────────────────────────────────────────
    det_pub_ = create_publisher<vision_msgs::msg::Detection2DArray>(
        "/detections/camera", 10);

    if (publish_viz_)
        viz_pub_ = create_publisher<sensor_msgs::msg::Image>(
            "/camera/annotated", 10);

    if (publish_bev_)
        bev_pub_ = create_publisher<sensor_msgs::msg::Image>(
            "/camera/bev", 10);

    if (publish_markers_)
        marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
            "/detections/camera_markers", 10);

    // ── Subscriber ────────────────────────────────────────────────────────────
    // Explicit RELIABLE QoS — publisher must match
    // (Python: QoSProfile(reliability=ReliabilityPolicy.RELIABLE, depth=10))
    auto qos = rclcpp::QoS(rclcpp::KeepLast(10))
                   .reliability(rclcpp::ReliabilityPolicy::Reliable);

    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "/camera/image_raw", qos,
        std::bind(&CameraNode::imageCallback, this, std::placeholders::_1));

    RCLCPP_INFO(get_logger(), "CameraNode ready. Listening on /camera/image_raw");
}

// ── Image callback ────────────────────────────────────────────────────────────

void CameraNode::imageCallback(
    const sensor_msgs::msg::Image::ConstSharedPtr& msg)
{
    // Convert ROS image to OpenCV
    cv_bridge::CvImageConstPtr cv_img;
    try {
        cv_img = cv_bridge::toCvShare(msg, "bgr8");
    } catch (const cv_bridge::Exception& e) {
        RCLCPP_WARN(get_logger(), "cv_bridge exception: %s", e.what());
        return;
    }

    const cv::Mat& frame = cv_img->image;
    if (frame.empty()) return;

    profiler_->frameStart();

    // ── TRT Inference ─────────────────────────────────────────────────────────
    float* output = engine_->infer(frame);
    auto   dets   = decoder_->decode(
        output, engine_->outputSize(), frame.cols, frame.rows);

    profiler_->frameEnd();
    profiler_->addInferMs(engine_->lastInferMs());
    profiler_->addPreprocessMs(engine_->lastPreprocessMs());
    profiler_->addTRTMs(engine_->lastTRTMs());

    // ── Publish detections ────────────────────────────────────────────────────
    auto det_msg = toRosMsg(dets, msg->header);
    det_pub_->publish(det_msg);

    // ── Publish annotated image ───────────────────────────────────────────────
    if (publish_viz_ && viz_pub_) {
        // Filter out barriers from visualization — too noisy in BEV and not useful to show in image
        std::vector<Detection> filtered_dets;
        for (const auto& d : dets) {
            if (d.class_name != "barrier") filtered_dets.push_back(d);
        }
        cv::Mat annotated = decoder_->draw(frame, filtered_dets);
        std::string fps_str = "FPS: " +
            std::to_string(static_cast<int>(profiler_->meanFPS())) +
            "  TRT: " +
            std::to_string(static_cast<int>(engine_->lastTRTMs())) + "ms";
        cv::putText(annotated, fps_str,
                    cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

        auto viz_msg = cv_bridge::CvImage(
            msg->header, "bgr8", annotated).toImageMsg();
        viz_pub_->publish(*viz_msg);
    }

    // ── Publish BEV ──────────────────────────────────────────────────────────
    if (publish_bev_ && bev_ && bev_pub_) {
        cv::Mat bev_img = bev_->render(dets, frame.cols, frame.rows);
        auto bev_msg = cv_bridge::CvImage(
            msg->header, "bgr8", bev_img).toImageMsg();
        bev_pub_->publish(*bev_msg);
    }

    // ── Publish markers ───────────────────────────────────────────────────────
    if (publish_markers_ && marker_pub_)
        publishMarkers(dets, msg->header, frame.cols, frame.rows);
    frame_count_++;
    if (frame_count_ % 30 == 0) {
        RCLCPP_INFO(get_logger(),
            "FPS: %.1f  Pre: %.1fms  TRT: %.1fms  Dets: %zu",
            profiler_->meanFPS(),
            profiler_->meanPreprocessMs(),
            profiler_->meanTRTMs(),
            dets.size());
    }
}

// ── Convert to ROS2 message ───────────────────────────────────────────────────

vision_msgs::msg::Detection2DArray CameraNode::toRosMsg(
    const std::vector<Detection>& dets,
    const std_msgs::msg::Header& header)
{
    vision_msgs::msg::Detection2DArray msg;
    msg.header = header;

    for (const auto& d : dets) {
        // Skip barriers
        if (d.class_name == "barrier") continue;

        vision_msgs::msg::Detection2D det;
        det.header = header;

        // Bounding box center + size
        det.bbox.center.position.x = (d.x1 + d.x2) * 0.5;
        det.bbox.center.position.y = (d.y1 + d.y2) * 0.5;
        det.bbox.size_x = d.x2 - d.x1;
        det.bbox.size_y = d.y2 - d.y1;

        // Class + score
        vision_msgs::msg::ObjectHypothesisWithPose hyp;
        hyp.hypothesis.class_id = d.class_name;
        hyp.hypothesis.score    = d.score;
        det.results.push_back(hyp);

        msg.detections.push_back(det);
    }

    return msg;
}

} // namespace edgedrive

// ── publishMarkers ────────────────────────────────────────────────────────────

namespace edgedrive {

void CameraNode::publishMarkers(
    const std::vector<Detection>& dets,
    const std_msgs::msg::Header& header,
    int img_w, int img_h)
{
    // Auto-compute principal point if not set
    float cx = (cx_ > 0) ? cx_ : img_w * 0.5f;
    float cy = (cy_ > 0) ? cy_ : img_h * 0.5f;

    visualization_msgs::msg::MarkerArray marker_array;

    // Delete all previous markers first (clean slate each frame)
    visualization_msgs::msg::Marker delete_marker;
    delete_marker.header = header;
    delete_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    marker_array.markers.push_back(delete_marker);

    int id = 0;
    for (const auto& det : dets) {
        // Skip barriers — too noisy for BEV visualization
        if (det.class_name == "barrier") continue;

        // Bottom-center of bounding box = ground contact point
        float u = (det.x1 + det.x2) * 0.5f;
        float v =  det.y2;

        float dv = v - cy;
        if (dv <= 1.0f) continue;  // above horizon

        // Ground plane projection
        float Z = camera_height_ * fy_ / dv;   // forward (meters)
        float X = (u - cx) * Z / fx_;           // lateral (meters)

        if (Z < 0.5f || Z > 30.0f) continue;

        // ── Cylinder marker (detection footprint) ────────────────────────────
        visualization_msgs::msg::Marker cyl;
        cyl.header    = header;
        cyl.header.frame_id = "cam_front";
        cyl.ns        = "camera_detections";
        cyl.id        = id++;
        cyl.type      = visualization_msgs::msg::Marker::CYLINDER;
        cyl.action    = visualization_msgs::msg::Marker::ADD;

        // Position in camera frame: X=lateral, Y=up(unused), Z=forward
        cyl.pose.position.x  = Z;   // forward = x in RViz2 (ego forward)
        cyl.pose.position.y  = -X;  // lateral (right = negative in cam frame)
        cyl.pose.position.z  = 0.0; // ground plane

        cyl.pose.orientation.w = 1.0;

        // Size: width estimated from box pixel width
        float box_w_px = det.x2 - det.x1;
        float est_w_m  = std::max(0.5f, std::min(3.5f,
                            box_w_px * Z / fx_));
        cyl.scale.x = est_w_m;
        cyl.scale.y = est_w_m;
        cyl.scale.z = det.score * 2.0f;  // height = confidence (max 2m)

        // Single green color for all camera detections
        // (red=fused, green=camera, blue=lidar in the fusion view)
        cyl.color.r = 0.1f;
        cyl.color.g = 0.9f;
        cyl.color.b = 0.1f;
        cyl.color.a = 0.7f;

        cyl.lifetime = rclcpp::Duration::from_seconds(1.0);
        marker_array.markers.push_back(cyl);

        // ── Text label ───────────────────────────────────────────────────────
        visualization_msgs::msg::Marker txt;
        txt.header    = header;
        txt.header.frame_id = "cam_front";
        txt.ns        = "camera_labels";
        txt.id        = id++;
        txt.type      = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        txt.action    = visualization_msgs::msg::Marker::ADD;

        txt.pose.position.x = Z;
        txt.pose.position.y = -X;
        txt.pose.position.z = cyl.scale.z + 0.3f;
        txt.pose.orientation.w = 1.0;

        txt.scale.z = 0.4f;  // text height in meters
        txt.color.r = txt.color.g = txt.color.b = 1.0f;
        txt.color.a = 1.0f;

        std::ostringstream ss;
        ss << det.class_name << "\n"
           << std::fixed << std::setprecision(1) << Z << "m";
        txt.text = ss.str();
        txt.lifetime = rclcpp::Duration::from_seconds(1.0);
        marker_array.markers.push_back(txt);
    }

    marker_pub_->publish(marker_array);
}

} // namespace edgedrive

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<edgedrive::CameraNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
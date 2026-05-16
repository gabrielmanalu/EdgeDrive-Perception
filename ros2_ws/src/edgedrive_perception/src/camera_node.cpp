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
#include <cv_bridge/cv_bridge.h>
#include <chrono>

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
    camera_height_    = declare_parameter("camera_height", 1.2f);

    RCLCPP_INFO(get_logger(), "EdgeDrive CameraNode starting...");
    RCLCPP_INFO(get_logger(), "  Engine    : %s", engine_path_.c_str());
    RCLCPP_INFO(get_logger(), "  Threshold : %.2f", score_threshold_);
    RCLCPP_INFO(get_logger(), "  Publish viz: %s", publish_viz_ ? "yes" : "no");
    RCLCPP_INFO(get_logger(), "  Publish BEV: %s", publish_bev_ ? "yes" : "no");

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

    // image_transport requires shared_from_this() — use a timer to defer
    // until after the constructor returns and shared_ptr is valid
    init_timer_ = create_wall_timer(
        std::chrono::milliseconds(0),
        [this]() {
            init_timer_->cancel();

            auto img_transport = std::make_shared<image_transport::ImageTransport>(
                shared_from_this());

            if (publish_viz_)
                viz_pub_ = img_transport->advertise("/camera/annotated", 1);

            if (publish_bev_)
                bev_pub_ = img_transport->advertise("/camera/bev", 1);

            image_sub_ = img_transport->subscribe(
                "/camera/image_raw", 1,
                std::bind(&CameraNode::imageCallback, this,
                          std::placeholders::_1));

            // Keep transport alive
            img_transport_ = img_transport;

            RCLCPP_INFO(get_logger(),
                "CameraNode ready. Listening on /camera/image_raw");
        });
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
    if (publish_viz_ && viz_pub_.getNumSubscribers() > 0) {
        cv::Mat annotated = decoder_->draw(frame, dets);

        // FPS overlay
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
        viz_pub_.publish(viz_msg);
    }

    // ── Publish BEV ──────────────────────────────────────────────────────────
    if (publish_bev_ && bev_ && bev_pub_.getNumSubscribers() > 0) {
        cv::Mat bev_img = bev_->render(dets, frame.cols, frame.rows);
        auto bev_msg = cv_bridge::CvImage(
            msg->header, "bgr8", bev_img).toImageMsg();
        bev_pub_.publish(bev_msg);
    }

    // ── Log stats every 30 frames ─────────────────────────────────────────────
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

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<edgedrive::CameraNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
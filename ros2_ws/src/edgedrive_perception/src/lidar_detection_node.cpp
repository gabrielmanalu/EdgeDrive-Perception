/*
 * lidar_detection_node.cpp — EdgeDrive Perception: CUDA-PointPillars ROS2 Node
 * ==============================================================================
 *
 * Subscribes to PointCloud2, runs CUDA-PointPillars TRT inference,
 * publishes Detection3DArray + MarkerArray (blue cylinders in RViz2).
 *
 * PointCloud2 conversion:
 *   nuScenes LiDAR has 5 fields (x,y,z,intensity,ring) at 20 bytes/point.
 *   CUDA-PointPillars expects 4 floats/point (x,y,z,intensity).
 *   Ring channel is stripped during conversion.
 */

#include "edgedrive_perception/lidar_detection_node.hpp"

#include <vision_msgs/msg/detection3_d.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <sensor_msgs/point_cloud2_iterator.hpp>

// CUDA-PointPillars headers
#include "pointpillar/pointpillar.hpp"
#include "pointpillar/lidar-postprocess.hpp"

#include <cuda_runtime.h>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace edgedrive {

// nuScenes class names — actual mmdetection3d ordering
// Verified against inference_detector output:
// cls=0→car, cls=7→pedestrian
static const std::vector<std::string> NUSCENES_CLASSES = {
    "car",                  // 0
    "truck",                // 1
    "construction_vehicle", // 2
    "bus",                  // 3
    "trailer",              // 4
    "barrier",              // 5
    "motorcycle",           // 6
    "pedestrian",           // 7
    "bicycle",              // 8
    "traffic_cone"          // 9
};

// Blue cylinder color for LiDAR detections in RViz2
// (contrasts with camera's green cylinders)
static const std::array<float, 3> LIDAR_COLOR = {0.2f, 0.6f, 1.0f};

// ── Constructor ───────────────────────────────────────────────────────────────

LidarDetectionNode::LidarDetectionNode(const rclcpp::NodeOptions& options)
    : Node("lidar_detection_node", options)
{
    // Parameters
    engine_path_      = declare_parameter("engine_path",
        "/workspace/cuda-pointpillars/model/pointpillar_fpn3.plan");
    score_threshold_  = declare_parameter("score_threshold", 0.15f);
    publish_markers_  = declare_parameter("publish_markers", true);

    RCLCPP_INFO(get_logger(), "LidarDetectionNode starting...");
    RCLCPP_INFO(get_logger(), "  Engine    : %s", engine_path_.c_str());
    RCLCPP_INFO(get_logger(), "  Threshold : %.2f", score_threshold_);
    RCLCPP_INFO(get_logger(), "  Markers   : %s", publish_markers_ ? "yes" : "no");

    // ── Create CUDA stream ────────────────────────────────────────────────────
    cudaStreamCreate(&stream_);

    // ── Create CUDA-PointPillars core ─────────────────────────────────────────
    pointpillar::lidar::VoxelizationParameter vp;
    vp.min_range = nvtype::Float3(-50.0f, -50.0f, -5.0f);
    vp.max_range = nvtype::Float3( 50.0f,  50.0f,  3.0f);
    vp.voxel_size = nvtype::Float3(0.25f, 0.25f, 8.0f);
    vp.grid_size  = vp.compute_grid_size(vp.max_range, vp.min_range, vp.voxel_size);
    vp.max_voxels = 30000;
    vp.max_points_per_voxel = 32;
    vp.max_points = 300000;
    vp.num_feature = 4;  // x, y, z, intensity (ring stripped)

    pointpillar::lidar::PostProcessParameter pp;
    pp.min_range     = vp.min_range;
    pp.max_range     = vp.max_range;
    pp.feature_size  = nvtype::Int2(vp.grid_size.x/2, vp.grid_size.y/2);
    pp.num_classes   = 10;
    pp.num_anchors   = 8;
    pp.len_per_anchor = 4;
    float anchors[32] = {
        2.60f, 0.87f, 1.0f, 0.0f,      // car rot=0
        2.60f, 0.87f, 1.0f, 1.5708f,   // car rot=90
        1.73f, 0.58f, 1.0f, 0.0f,      // pedestrian rot=0
        1.73f, 0.58f, 1.0f, 1.5708f,   // pedestrian rot=90
        1.0f,  1.0f,  1.0f, 0.0f,      // bicycle rot=0
        1.0f,  1.0f,  1.0f, 1.5708f,   // bicycle rot=90
        0.4f,  0.4f,  1.0f, 0.0f,      // cone/barrier rot=0
        0.4f,  0.4f,  1.0f, 1.5708f,   // cone/barrier rot=90
    };
    memcpy(pp.anchors, anchors, sizeof(anchors));
    pp.anchor_bottom_heights = nvtype::Float3(-1.8f, -1.8f, -1.8f);
    pp.num_box_values = 9;
    pp.score_thresh   = score_threshold_;
    pp.nms_thresh     = 0.2f;
    pp.dir_offset     = -0.7854f;

    pointpillar::lidar::CoreParameter param;
    param.voxelization = vp;
    param.lidar_model  = engine_path_;
    param.lidar_post   = pp;

    pp_core_ = pointpillar::lidar::create_core(param);
    if (!pp_core_) {
        RCLCPP_FATAL(get_logger(), "Failed to create PointPillars core. "
                     "Check engine path: %s", engine_path_.c_str());
        throw std::runtime_error("PointPillars init failed");
    }

    RCLCPP_INFO(get_logger(), "PointPillars engine loaded.");
    pp_core_->print();

    // ── Publishers ────────────────────────────────────────────────────────────
    det_pub_ = create_publisher<vision_msgs::msg::Detection3DArray>(
        "/detections/lidar", 10);

    if (publish_markers_)
        marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
            "/detections/lidar_markers", 10);

    // ── Subscriber ────────────────────────────────────────────────────────────
    auto qos = rclcpp::QoS(rclcpp::KeepLast(10))
                   .reliability(rclcpp::ReliabilityPolicy::Reliable);

    lidar_sub_ = create_subscription<sensor_msgs::msg::PointCloud2>(
        "/nuscenes/lidar/pointcloud", qos,
        std::bind(&LidarDetectionNode::lidarCallback, this,
                  std::placeholders::_1));

    RCLCPP_INFO(get_logger(),
        "LidarDetectionNode ready. Listening on /nuscenes/lidar/pointcloud");
}

LidarDetectionNode::~LidarDetectionNode() {
    if (stream_) cudaStreamDestroy(stream_);
}

// ── Lidar callback ────────────────────────────────────────────────────────────

void LidarDetectionNode::lidarCallback(
    const sensor_msgs::msg::PointCloud2::ConstSharedPtr& msg)
{
    // nuScenes PointCloud2: fields x,y,z,intensity,ring — 5 floats/point (20 bytes)
    // CUDA-PointPillars expects: x,y,z,intensity — 4 floats/point (16 bytes)
    // Strip ring channel during conversion

    const int src_fields = msg->point_step / sizeof(float);  // 5 for nuScenes
    const int num_points = msg->width * msg->height;
    const int dst_fields = 4;  // x,y,z,intensity only

    // Build stripped float array
    std::vector<float> points(num_points * dst_fields);
    const float* src = reinterpret_cast<const float*>(msg->data.data());

    for (int i = 0; i < num_points; ++i) {
        points[i * dst_fields + 0] = src[i * src_fields + 0];  // x
        points[i * dst_fields + 1] = src[i * src_fields + 1];  // y
        points[i * dst_fields + 2] = src[i * src_fields + 2];  // z
        points[i * dst_fields + 3] = src[i * src_fields + 3];  // intensity
    }

    // Run CUDA-PointPillars inference
    auto boxes = pp_core_->forward(points.data(), num_points, stream_);
    cudaStreamSynchronize(stream_);

    // Filter by score threshold
    std::vector<pointpillar::lidar::BoundingBox> filtered;
    for (const auto& b : boxes)
        if (b.score >= score_threshold_) filtered.push_back(b);

    // Publish Detection3DArray
    vision_msgs::msg::Detection3DArray det_msg;
    det_msg.header = msg->header;
    det_msg.header.frame_id = "base_link";

    for (const auto& b : filtered) {
        vision_msgs::msg::Detection3D det;
        det.header = det_msg.header;
        // nuScenes LiDAR→ego: ego_forward = +LiDAR_Y, ego_left = -LiDAR_X
        det.bbox.center.position.x =  b.y;  // ego forward
        det.bbox.center.position.y = -b.x;  // ego left
        det.bbox.center.position.z =  b.z;
        det.bbox.size.x = b.w;
        det.bbox.size.y = b.l;
        det.bbox.size.z = b.h;

        vision_msgs::msg::ObjectHypothesisWithPose hyp;
        std::string cls = (b.id >= 0 && b.id < (int)NUSCENES_CLASSES.size())
                          ? NUSCENES_CLASSES[b.id] : "unknown";
        hyp.hypothesis.class_id = cls;
        hyp.hypothesis.score    = b.score;
        det.results.push_back(hyp);
        det_msg.detections.push_back(det);
    }
    det_pub_->publish(det_msg);

    // Publish MarkerArray
    if (publish_markers_ && marker_pub_)
        publishMarkers(filtered, det_msg.header);

    frame_count_++;
    if (frame_count_ % 10 == 0) {
        RCLCPP_INFO(get_logger(),
            "LiDAR frame %d: %zu detections (%d points)",
            frame_count_, filtered.size(), num_points);
    }
}

// ── MarkerArray publisher ─────────────────────────────────────────────────────

void LidarDetectionNode::publishMarkers(
    const std::vector<pointpillar::lidar::BoundingBox>& boxes,
    const std_msgs::msg::Header& header)
{
    visualization_msgs::msg::MarkerArray marker_array;

    // Delete previous markers
    visualization_msgs::msg::Marker del;
    del.header = header;
    del.header.frame_id = "base_link";
    del.action = visualization_msgs::msg::Marker::DELETEALL;
    marker_array.markers.push_back(del);

    int id = 0;
    for (const auto& b : boxes) {
        // ── Cylinder marker ───────────────────────────────────────────────────
        visualization_msgs::msg::Marker cyl;
        cyl.header    = header;
        cyl.header.frame_id = "base_link";
        cyl.ns        = "lidar_detections";
        cyl.id        = id++;
        cyl.type      = visualization_msgs::msg::Marker::CYLINDER;
        cyl.action    = visualization_msgs::msg::Marker::ADD;

        // Directly map coordinates since network output is now base_link aligned
        cyl.pose.position.x  = b.x;
        cyl.pose.position.y  = b.y;
        cyl.pose.position.z  = 0.0;
        cyl.pose.orientation.w = 1.0;

        cyl.scale.x = std::max(0.5f, b.w);
        cyl.scale.y = std::max(0.5f, b.l);
        cyl.scale.z = b.score * 2.0f;

        cyl.color.r = LIDAR_COLOR[0];
        cyl.color.g = LIDAR_COLOR[1];
        cyl.color.b = LIDAR_COLOR[2];
        cyl.color.a = 0.7f;
        cyl.lifetime = rclcpp::Duration::from_seconds(1.0);
        marker_array.markers.push_back(cyl);

        // ── Text label ────────────────────────────────────────────────────────
        visualization_msgs::msg::Marker txt;
        txt.header    = header;
        txt.header.frame_id = "base_link";
        txt.ns        = "lidar_labels";
        txt.id        = id++;
        txt.type      = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        txt.action    = visualization_msgs::msg::Marker::ADD;

        txt.pose.position.x = b.x;
        txt.pose.position.y = b.y;
        txt.pose.position.z = cyl.scale.z + 0.3f;
        txt.pose.orientation.w = 1.0;
        txt.scale.z = 0.4f;
        txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0f;
        txt.lifetime = rclcpp::Duration::from_seconds(1.0);

        std::string cls = (b.id >= 0 && b.id < (int)NUSCENES_CLASSES.size())
                          ? NUSCENES_CLASSES[b.id] : "unknown";
        float dist = std::sqrt(b.x*b.x + b.y*b.y);
        std::ostringstream ss;
        ss << cls << "\n" << std::fixed << std::setprecision(1) << dist << "m";
        txt.text = ss.str();
        marker_array.markers.push_back(txt);
    }

    marker_pub_->publish(marker_array);
}

} // namespace edgedrive

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<edgedrive::LidarDetectionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
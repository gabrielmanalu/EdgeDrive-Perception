/*
 * fusion_node.cpp — EdgeDrive Perception: Camera-LiDAR Late Fusion
 * ==================================================================
 *
 * ApproximateTime sync → BEV projection → Hungarian matching → MarkerArray
 */

#include "edgedrive_perception/fusion_node.hpp"

#include <vision_msgs/msg/detection3_d.hpp>
#include <vision_msgs/msg/object_hypothesis_with_pose.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <cmath>
#include <limits>
#include <sstream>
#include <iomanip>

namespace edgedrive {

// ── Constructor ───────────────────────────────────────────────────────────────

FusionNode::FusionNode(const rclcpp::NodeOptions& options)
    : Node("fusion_node", options)
{
    match_threshold_ = declare_parameter("match_threshold", 3.0f);
    camera_height_   = declare_parameter("camera_height",   1.2f);
    fx_              = declare_parameter("camera_fx",       640.0f);
    fy_              = declare_parameter("camera_fy",       640.0f);
    cx_              = declare_parameter("camera_cx",       0.0f);
    cy_              = declare_parameter("camera_cy",       0.0f);
    double sync_tol  = declare_parameter("sync_tolerance",  0.3);

    RCLCPP_INFO(get_logger(), "FusionNode starting...");
    RCLCPP_INFO(get_logger(), "  Match threshold : %.1fm", match_threshold_);
    RCLCPP_INFO(get_logger(), "  Sync tolerance  : %.2fs", sync_tol);

    // ── Publishers ────────────────────────────────────────────────────────────
    fused_pub_ = create_publisher<vision_msgs::msg::Detection3DArray>(
        "/detections/fused", 10);
    marker_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
        "/visualization/fused", 10);

    // ── ApproximateTime sync subscribers ──────────────────────────────────────
    auto qos = rclcpp::QoS(rclcpp::KeepLast(10))
                   .reliability(rclcpp::ReliabilityPolicy::Reliable);

    cam_sub_.subscribe(this, "/detections/camera", qos.get_rmw_qos_profile());
    lidar_sub_.subscribe(this, "/detections/lidar", qos.get_rmw_qos_profile());

    sync_ = std::make_shared<Synchronizer>(
        SyncPolicy(10), cam_sub_, lidar_sub_);
    sync_->setMaxIntervalDuration(
        rclcpp::Duration::from_seconds(sync_tol));
    sync_->registerCallback(
        std::bind(&FusionNode::fusionCallback, this,
                  std::placeholders::_1, std::placeholders::_2));

    RCLCPP_INFO(get_logger(),
        "FusionNode ready. Waiting for /detections/camera + /detections/lidar");
}

// ── Fusion callback ───────────────────────────────────────────────────────────

void FusionNode::fusionCallback(
    const CamMsg::ConstSharedPtr&   cam_msg,
    const LidarMsg::ConstSharedPtr& lidar_msg)
{
    frame_count_++;

    // Use a reasonable image size for nuScenes front camera
    const int img_w = 1600, img_h = 900;

    // Project to BEV
    auto cam_bev   = cameraDetsToBEV(*cam_msg, img_w, img_h);
    auto lidar_bev = lidarDetsToBEV(*lidar_msg);

    // Hungarian matching
    auto assignment = hungarianMatch(cam_bev, lidar_bev);

    // Build fused Detection3DArray
    vision_msgs::msg::Detection3DArray fused_msg;
    fused_msg.header = lidar_msg->header;
    fused_msg.header.frame_id = "base_link";

    // Matched: camera + LiDAR
    for (int ci = 0; ci < (int)assignment.size(); ci++) {
        int li = assignment[ci];
        if (li < 0) continue;

        const auto& lidar_det = lidar_msg->detections[lidar_bev[li].orig_idx];
        vision_msgs::msg::Detection3D det = lidar_det;

        // Boost score with camera confidence
        float cam_score   = cam_bev[ci].score;
        float lidar_score = lidar_bev[li].score;
        float fused_score = 0.6f * lidar_score + 0.4f * cam_score;

        if (!det.results.empty()) {
            det.results[0].hypothesis.score = fused_score;
            // Use camera class if lidar class is uncertain
            if (lidar_score < 0.3f && !cam_bev[ci].class_id.empty())
                det.results[0].hypothesis.class_id = cam_bev[ci].class_id;
        }
        fused_msg.detections.push_back(det);
    }

    // Unmatched LiDAR
    std::vector<bool> lidar_matched(lidar_bev.size(), false);
    for (int ci = 0; ci < (int)assignment.size(); ci++)
        if (assignment[ci] >= 0) lidar_matched[assignment[ci]] = true;

    for (int li = 0; li < (int)lidar_bev.size(); li++) {
        if (!lidar_matched[li])
            fused_msg.detections.push_back(
                lidar_msg->detections[lidar_bev[li].orig_idx]);
    }

    fused_pub_->publish(fused_msg);
    publishMarkers(cam_bev, lidar_bev, assignment, fused_msg.header);

    if (frame_count_ % 5 == 0) {
        int matched = 0;
        for (auto a : assignment) if (a >= 0) matched++;
        RCLCPP_INFO(get_logger(),
            "Frame %d: cam=%zu lidar=%zu matched=%d fused=%zu",
            frame_count_,
            cam_bev.size(), lidar_bev.size(),
            matched, fused_msg.detections.size());
    }
}

// ── Camera detections → BEV ───────────────────────────────────────────────────

std::vector<BEVPoint> FusionNode::cameraDetsToBEV(
    const CamMsg& msg, int img_w, int img_h)
{
    float cx = (cx_ > 0) ? cx_ : img_w * 0.5f;
    float cy = (cy_ > 0) ? cy_ : img_h * 0.5f;

    std::vector<BEVPoint> result;
    for (int i = 0; i < (int)msg.detections.size(); i++) {
        const auto& det = msg.detections[i];

        // Skip barriers — not useful for fusion
        if (!det.results.empty() &&
            det.results[0].hypothesis.class_id == "barrier") continue;

        float u = det.bbox.center.position.x;
        float v = det.bbox.center.position.y + det.bbox.size_y * 0.5f; // bottom

        float dv = v - cy;
        if (dv <= 1.0f) continue;

        float Z = camera_height_ * fy_ / dv;
        float X = (u - cx) * Z / fx_;

        if (Z < 1.0f || Z > 60.0f) continue;

        float score = 0.0f;
        std::string cls;
        if (!det.results.empty()) {
            score = det.results[0].hypothesis.score;
            cls   = det.results[0].hypothesis.class_id;
        }

        result.push_back({Z, -X, i, score, cls});
    }
    return result;
}

// ── LiDAR detections → BEV ───────────────────────────────────────────────────

std::vector<BEVPoint> FusionNode::lidarDetsToBEV(const LidarMsg& msg)
{
    std::vector<BEVPoint> result;
    for (int i = 0; i < (int)msg.detections.size(); i++) {
        const auto& det = msg.detections[i];

        // lidar_detection_node already converts to ego frame:
        // position.x = ego forward, position.y = ego left
        float x = det.bbox.center.position.x;
        float y = det.bbox.center.position.y;

        float score = 0.0f;
        std::string cls;
        if (!det.results.empty()) {
            score = det.results[0].hypothesis.score;
            cls   = det.results[0].hypothesis.class_id;
        }
        result.push_back({x, y, i, score, cls});
    }
    return result;
}

// ── Hungarian algorithm ───────────────────────────────────────────────────────

std::vector<int> FusionNode::hungarianMatch(
    const std::vector<BEVPoint>& cam,
    const std::vector<BEVPoint>& lidar)
{
    int n_cam   = cam.size();
    int n_lidar = lidar.size();
    std::vector<int> assignment(n_cam, -1);

    if (n_cam == 0 || n_lidar == 0) return assignment;

    // Build cost matrix [n_cam × n_lidar]
    std::vector<std::vector<float>> cost(n_cam,
        std::vector<float>(n_lidar, std::numeric_limits<float>::max()));

    for (int i = 0; i < n_cam; i++) {
        for (int j = 0; j < n_lidar; j++) {
            // Angular matching — compare bearing angle, not absolute BEV distance
            // Monocular depth is unreliable; angle from camera is accurate
            float cam_angle   = std::atan2(cam[i].y,   cam[i].x);
            float lidar_angle = std::atan2(lidar[j].y, lidar[j].x);
            float angle_diff  = std::abs(cam_angle - lidar_angle);
            // Wrap to [-pi, pi]
            if (angle_diff > M_PI) angle_diff = 2.0f * M_PI - angle_diff;

            // Convert threshold from meters to radians at ~20m range
            float angular_threshold = std::atan2(match_threshold_, 20.0f);
            if (angle_diff <= angular_threshold) {
                // Weight by BEV distance but allow angular matches
                float dx = cam[i].x - lidar[j].x;
                float dy = cam[i].y - lidar[j].y;
                float d  = std::sqrt(dx*dx + dy*dy);
                cost[i][j] = angle_diff * 20.0f;  // normalize to meters-equivalent
            }
        }
    }

    // Greedy assignment (sufficient for small detection counts)
    // TODO: replace with full Hungarian for large counts
    std::vector<bool> lidar_taken(n_lidar, false);

    // Sort camera detections by number of valid lidar candidates
    std::vector<int> cam_order(n_cam);
    std::iota(cam_order.begin(), cam_order.end(), 0);

    for (int i : cam_order) {
        float   best_cost = std::numeric_limits<float>::max();
        int     best_j    = -1;

        for (int j = 0; j < n_lidar; j++) {
            if (!lidar_taken[j] && cost[i][j] < best_cost) {
                best_cost = cost[i][j];
                best_j    = j;
            }
        }
        if (best_j >= 0) {
            assignment[i]        = best_j;
            lidar_taken[best_j]  = true;
        }
    }

    return assignment;
}

// ── MarkerArray publisher ─────────────────────────────────────────────────────

void FusionNode::publishMarkers(
    const std::vector<BEVPoint>& cam_bev,
    const std::vector<BEVPoint>& lidar_bev,
    const std::vector<int>&      assignment,
    const std_msgs::msg::Header& header)
{
    visualization_msgs::msg::MarkerArray arr;

    // Delete previous
    visualization_msgs::msg::Marker del;
    del.header = header;
    del.action = visualization_msgs::msg::Marker::DELETEALL;
    arr.markers.push_back(del);

    int id = 0;

    // ── Ego vehicle marker (white box at origin) ──────────────────────────────
    visualization_msgs::msg::Marker ego;
    ego.header = header;
    ego.ns     = "ego";
    ego.id     = id++;
    ego.type   = visualization_msgs::msg::Marker::CUBE;
    ego.action = visualization_msgs::msg::Marker::ADD;
    ego.pose.position.x = 0.0;
    ego.pose.position.y = 0.0;
    ego.pose.position.z = 0.5;
    ego.pose.orientation.w = 1.0;
    ego.scale.x = 4.0;   // car length
    ego.scale.y = 1.8;   // car width
    ego.scale.z = 1.5;   // car height
    ego.color.r = ego.color.g = ego.color.b = 1.0f;
    ego.color.a = 0.9f;
    ego.lifetime = rclcpp::Duration::from_seconds(0);  // permanent
    arr.markers.push_back(ego);

    // Ego label
    visualization_msgs::msg::Marker ego_txt;
    ego_txt.header = header;
    ego_txt.ns     = "ego";
    ego_txt.id     = id++;
    ego_txt.type   = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
    ego_txt.action = visualization_msgs::msg::Marker::ADD;
    ego_txt.pose.position.x = 0.0;
    ego_txt.pose.position.y = 0.0;
    ego_txt.pose.position.z = 2.2;
    ego_txt.pose.orientation.w = 1.0;
    ego_txt.scale.z = 0.5f;
    ego_txt.color.r = ego_txt.color.g = ego_txt.color.b = ego_txt.color.a = 1.0f;
    ego_txt.text = "EGO";
    ego_txt.lifetime = rclcpp::Duration::from_seconds(0);
    arr.markers.push_back(ego_txt);

    // Forward direction arrow
    visualization_msgs::msg::Marker arrow;
    arrow.header = header;
    arrow.ns     = "ego";
    arrow.id     = id++;
    arrow.type   = visualization_msgs::msg::Marker::ARROW;
    arrow.action = visualization_msgs::msg::Marker::ADD;
    arrow.pose.position.x = 2.0;
    arrow.pose.position.y = 0.0;
    arrow.pose.position.z = 0.5;
    arrow.pose.orientation.w = 1.0;
    arrow.scale.x = 3.0;   // arrow length
    arrow.scale.y = 0.3;
    arrow.scale.z = 0.3;
    arrow.color.r = 1.0f; arrow.color.g = 1.0f; arrow.color.b = 0.0f;
    arrow.color.a = 0.9f;
    arrow.lifetime = rclcpp::Duration::from_seconds(0);
    arr.markers.push_back(arrow);

    // Colors
    // red   = fused  (0.9, 0.1, 0.1)
    // green = cam only (0.1, 0.9, 0.1)
    // blue  = lidar only (0.2, 0.6, 1.0)

    std::vector<bool> lidar_matched(lidar_bev.size(), false);

    auto make_cylinder = [&](float x, float y, float score,
                              float r, float g, float b,
                              const std::string& ns,
                              const std::string& label) {
        visualization_msgs::msg::Marker cyl;
        cyl.header    = header;
        cyl.ns        = ns;
        cyl.id        = id++;
        cyl.type      = visualization_msgs::msg::Marker::CYLINDER;
        cyl.action    = visualization_msgs::msg::Marker::ADD;
        cyl.pose.position.x  = x;
        cyl.pose.position.y  = y;
        cyl.pose.position.z  = 0.0;
        cyl.pose.orientation.w = 1.0;
        cyl.scale.x = 1.5f;
        cyl.scale.y = 1.5f;
        cyl.scale.z = std::max(0.3f, score * 2.5f);
        cyl.color.r = r; cyl.color.g = g; cyl.color.b = b;
        cyl.color.a = 0.8f;
        cyl.lifetime = rclcpp::Duration::from_seconds(1.0);
        arr.markers.push_back(cyl);

        // Text label
        visualization_msgs::msg::Marker txt;
        txt.header = header;
        txt.ns     = ns + "_txt";
        txt.id     = id++;
        txt.type   = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        txt.action = visualization_msgs::msg::Marker::ADD;
        txt.pose.position.x = x;
        txt.pose.position.y = y;
        txt.pose.position.z = cyl.scale.z + 0.4f;
        txt.pose.orientation.w = 1.0;
        txt.scale.z = 0.4f;
        txt.color.r = txt.color.g = txt.color.b = txt.color.a = 1.0f;
        txt.lifetime = rclcpp::Duration::from_seconds(1.0);
        std::ostringstream ss;
        float dist = std::sqrt(x*x + y*y);
        ss << label << "\n" << std::fixed << std::setprecision(1) << dist << "m";
        txt.text = ss.str();
        arr.markers.push_back(txt);
    };

    // Red — matched (fused)
    for (int ci = 0; ci < (int)assignment.size(); ci++) {
        int li = assignment[ci];
        if (li < 0) continue;
        lidar_matched[li] = true;

        float x = (cam_bev[ci].x + lidar_bev[li].x) * 0.5f;
        float y = (cam_bev[ci].y + lidar_bev[li].y) * 0.5f;
        float score = 0.6f * lidar_bev[li].score + 0.4f * cam_bev[ci].score;
        std::string lbl = lidar_bev[li].class_id.empty()
                        ? cam_bev[ci].class_id
                        : lidar_bev[li].class_id;
        make_cylinder(x, y, score, 0.9f, 0.1f, 0.1f, "fused", lbl);
    }

    // Green — camera only
    for (int ci = 0; ci < (int)assignment.size(); ci++) {
        if (assignment[ci] >= 0) continue;
        make_cylinder(cam_bev[ci].x, cam_bev[ci].y,
                      cam_bev[ci].score,
                      0.1f, 0.9f, 0.1f, "cam_only",
                      cam_bev[ci].class_id);
    }

    // Blue — lidar only
    for (int li = 0; li < (int)lidar_bev.size(); li++) {
        if (lidar_matched[li]) continue;
        make_cylinder(lidar_bev[li].x, lidar_bev[li].y,
                      lidar_bev[li].score,
                      0.2f, 0.6f, 1.0f, "lidar_only",
                      lidar_bev[li].class_id);
    }

    marker_pub_->publish(arr);
}

} // namespace edgedrive

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<edgedrive::FusionNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
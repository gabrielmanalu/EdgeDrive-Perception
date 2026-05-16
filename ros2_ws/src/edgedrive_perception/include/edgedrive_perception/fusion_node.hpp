#pragma once

/**
 * fusion_node.hpp — Camera-LiDAR Fusion ROS2 Node (placeholder)
 * ==============================================================
 *
 * Subscribes (synchronized):
 *   /camera/image_raw       (sensor_msgs/Image)
 *   /lidar/pointcloud       (sensor_msgs/PointCloud2)
 *
 * Runs:
 *   Camera branch  → YOLO26n TRT INT8 → 2D detections → BEV projection
 *   LiDAR branch   → CUDA-PointPillars → 3D boxes → BEV
 *   Fusion         → Hungarian matching in BEV space
 *
 * Publishes:
 *   /detections/fused    (vision_msgs/Detection3DArray)
 *   /visualization/fused (visualization_msgs/MarkerArray) → RViz2
 *
 * Sync policy:
 *   message_filters::sync_policies::ApproximateTime
 *   (camera @ 12Hz, LiDAR @ 20Hz — timestamps won't match exactly)
 *
 * Status: ⬜ planned — requires CUDA-PointPillars on Jetson
 *         Stub in place for colcon build verification.
 */

// See docs/sensor_fusion_analysis.md for fusion design.
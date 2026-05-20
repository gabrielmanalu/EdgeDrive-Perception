# ROS2 Architecture — EdgeDrive Perception

## Overview

The ROS2 package wraps the existing C++ TensorRT pipeline in a standard
ROS2 node architecture, enabling integration with other autonomous driving
components and standard ROS2 tooling (RViz2, rosbag2, ros2 topic, etc.)

```
nuScenes bag / USB camera / CSI camera
            │
            │ sensor_msgs/Image
            ▼
    ┌───────────────────┐
    │   camera_node     │  ← YOLO26n TRT INT8
    │                   │    202 FPS standalone
    │  trt_engine.cpp   │    ~180 FPS via ROS2 topics
    │  yolo26_decoder   │    (1600×900 nuScenes input)
    │  bev_visualizer   │
    └───────────────────┘
            │                    │                    │                    │
            │ Detection2DArray   │ Image (annotated)  │ Image (BEV)        │ MarkerArray
            ▼                    ▼                    ▼                    ▼
    /detections/camera   /camera/annotated       /camera/bev    /detections/camera_markers
                                                                      (RViz2 BEV cylinders)

nuScenes LiDAR bag
            │
            │ sensor_msgs/PointCloud2
            ▼
    ┌─────────────────────────┐
    │  CUDA-PointPillars      │  ← standalone C++ 
    │                         │    ~25-30ms, ~37-45 FPS
    │  lidar-voxelization.cu  │    43,440 pts/frame
    │  pillarScatterCHW kernel│    140-191 detections/frame
    │  TRT backbone FP16      │
    │  lidar-postprocess.cu   │
    └─────────────────────────┘
            │
            ▼ wrapped as ROS2 node 
    ┌─────────────────────────┐
    │  lidar_detection_node   │ 
    │                         │  ← subscribes /nuscenes/lidar/pointcloud
    │  links libpointpillar   │    publishes /detections/lidar_markers
    │  _core.so               │    blue cylinders in RViz2
    └─────────────────────────┘
            │
            ▼
    ┌───────────────────┐
    │   fusion_node     │  ← Hungarian matching
    │                   │    ApproximateTime sync
    │  late_fusion.cpp  │    BEV projection
    └───────────────────┘
            │                         │
            │ Detection3DArray        │ MarkerArray
            ▼                         ▼
    /detections/fused         /visualization/fused → RViz2
```

---

## Container Architecture

```
Host (Jetson Orin Nano Super, JetPack R36.4.7):
  CUDA 12.6, TRT 10.3.0, OpenCV 4.8 (system install)

edgedrive-ros2 container:
  Base: l4t-jetpack:r36.4.0
  Adds: ROS2 Humble Desktop, colcon, cv_bridge, vision_msgs
  TRT : 10.3.0 (same as host → engines compatible, no rebuild)
  Binary compiled at docker build time → lives inside container

edgedrive container (production):
  Base: l4t-jetpack:r36.4.0
  Adds: GStreamer only
  No ROS2 → minimal, fastest startup
```

---

## Topic Graph

### Current

```
[rosbag2 player]──/nuscenes/camera/image_raw──►[camera_node]
                                                     │
                                    ┌────────────────┤────────────────────┐
                                    │                │                    │
                          /detections/camera  /camera/annotated  /detections/camera_markers
                                    │                                     │
                              (Detection2DArray)                    (MarkerArray)
                              header, detections[]:               green cylinders
                                bbox.center.x/y                   in RViz2 BEV
                                bbox.size_x/y
                                results[0].class_id
                                results[0].score

[rosbag2 player]──/nuscenes/lidar/pointcloud ─►[CUDA-PointPillars C++ 
                                                    ─► lidar_detection_node]
                                                     │
                                                     │
                        ┌────────────────────────────┤
                        │                            │
             /detections/lidar          /detections/lidar_markers
             (Detection3DArray)          (blue cylinders → RViz2)
             x,y,z,w,l,h,yaw,cls,score   25-30ms per frame
```

### Planned — fusion_node

```
[rosbag2 player]──/nuscenes/camera/image_raw──►[camera_node]──/detections/camera───►┐
                │                                                                   │
                └──/nuscenes/lidar/pointcloud──►[lidar_detection_node]              │
                                                           │                        │
                                                           ──►  /detections/lidar──►┤             
                                                                                    │
                                                                               [fusion_node]
                                                                                    │
                                                               ┌────────────────────┤
                                                               │                    │
                                                    /detections/fused    /visualization/fused
                                                    (Detection3DArray)   (MarkerArray → RViz2)
```

---

## QoS Configuration

ROS2 QoS mismatches cause silent message drops — no error, no warning.
All camera_node publishers and subscribers use:

```
Reliability : RELIABLE
Durability  : VOLATILE
History     : KEEP_LAST (depth 10)
```

Publishers must explicitly match all three policies. Example (Python):
```python
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)
```

---

## nuScenes Data Pipeline

```
/data/sets/nuscenes/          ← nuScenes Mini (3.88GB)
    samples/CAM_FRONT/        ← 1600×900 JPEG images
    samples/LIDAR_TOP/        ← .bin point cloud files
    v1.0-mini/                ← metadata JSON

scripts/nuscenes_to_ros2bag.py
    → reads nuScenes metadata + sensor data
    → writes rosbags .db3 format (version 8, ROS2 Humble compatible)
    → output: bags/nuscenes_scene0/ (187MB for 39 frames)

bags/nuscenes_scene0/
    ├── metadata.yaml          ← bag metadata
    └── nuscenes_scene0.db3    ← SQLite database with messages

ros2 bag play bags/nuscenes_scene0
    → /nuscenes/camera/image_raw  @ 2 Hz (nuScenes capture rate)
    → /nuscenes/lidar/pointcloud  @ 2 Hz (synchronized)
```

---

## Performance Analysis

| Condition | FPS | Pre | TRT | Note |
|---|---|---|---|---|
| Standalone C++ (720p) | 203 | 1.1ms | 3.7ms | No ROS2 overhead |
| ROS2 camera_node (720p) | ~195 | 1.1ms | 3.7ms | topic serialize overhead |
| ROS2 camera_node (1600×900) | ~180 | 1.8ms | 4.0ms | nuScenes resolution |
| CUDA-PointPillars (43K pts) | ~37-45 | 0.2ms vox | 13-17ms bb | - |
| lidar_detection_node (ROS2) | ~2 | — | ~25-30ms/frame | gated by bag 2Hz |

ROS2 overhead is ~5 FPS due to:
- `sensor_msgs/Image` serialization/deserialization per frame
- `cv_bridge::toCvShare` copy (zero-copy not yet implemented)
- DDS middleware latency

At 2 Hz bag replay rate, the node processes each frame as fast as possible
(~180 FPS) then waits for the next message. Mean FPS reflects burst speed.

---

## Known Issues

**`not found: /opt/ros/humble/install/local_setup.bash`**
Harmless warning from the entrypoint. `/opt/ros/humble/install/` doesn't
exist — ROS2 is installed directly to `/opt/ros/humble/`. No impact.

**Mounting full workspace overwrites container binary**
`-v ~/EdgeDrive-Perception:/workspace` hides the compiled binary.
Use compose services which mount only `weights/` and `bags/`.

**DDS discovery between separate containers**
Even with `--network host`, DDS multicast discovery sometimes fails
between two separate containers. Solution: run bag replay and camera_node
in the same container (as in `run_ros2_bag_demo.sh`).

**CUDA-PointPillars class labels may be inaccurate**
Our simplified 4-size anchor set doesn't fully match all nuScenes 10-class
anchors. Detection geometry and count are correct — class assignment will
be refined when fusion_node is wired up. See `cuda-pointpillars-patches/`.
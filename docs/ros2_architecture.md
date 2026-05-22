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
    │  bev_visualizer   │    barriers filtered from all outputs
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
    │  CUDA-PointPillars FPN3 │  ← standalone C++
    │                         │    ~25-30ms, ~37-45 FPS
    │  lidar-voxelization.cu  │    43,440 pts/frame
    │  pillarScatterCHW kernel│    140-191 detections/frame
    │  TRT FPN3 backbone FP16 │    10 classes, 3 FPN levels
    │  lidar-postprocess.cu   │
    └─────────────────────────┘
            │
            ▼ wrapped as ROS2 node 
    ┌─────────────────────────┐
    │  lidar_detection_node   │ 
    │                         │  ← subscribes /nuscenes/lidar/pointcloud
    │  links libpointpillar   │    publishes /detections/lidar (Detection3DArray)
    │  _core.so               │    publishes /detections/lidar_markers (blue → RViz2)
    └─────────────────────────┘
            │
            ▼
    ┌───────────────────┐
    │   fusion_node     │     ← Hungarian BEV matching
    │                   │       ApproximateTime sync
    │  late_fusion.cpp  │       BEV projection + angular distance
    └───────────────────┘
            │                         │
            │ Detection3DArray        │ MarkerArray
            ▼                         ▼
    /detections/fused         /visualization/fused → RViz2
                                Red=fused | Green=cam | Blue=lidar
                                + live stats overlay (CAM/LiDAR/FUSED count)
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

```
[rosbag2 player]──/nuscenes/camera/image_raw──►[camera_node]
                                                     │
                                    ┌────────────────┤────────────────────┐
                                    │                │                    │
                          /detections/camera  /camera/annotated  /detections/camera_markers
                                    │                                     │
                              (Detection2DArray)                    (MarkerArray)
                              header, detections[]:               green cylinders
                                bbox.center.x/y  (pixels)         in RViz2 BEV
                                results[0].class_id
                                results[0].score
                                (barriers excluded)

[rosbag2 player]──/nuscenes/lidar/pointcloud──►[lidar_detection_node]
                                                     │
                        ┌────────────────────────────┤
                        │                            │
             /detections/lidar          /detections/lidar_markers
             (Detection3DArray)          (blue cylinders → RViz2)
             ego-frame x,y,z,w,l,h      25-30ms per frame
             cls,score (10 classes)     LiDAR→ego transform applied


[/detections/camera] ──►┐
                         ├──► [fusion_node] ──► /visualization/fused (MarkerArray → RViz2)
[/detections/lidar]  ──►┘         │             Red=fused | Green=cam | Blue=lidar
                                   │             Stats overlay: CAM/LiDAR/FUSED count
                                   └──► /detections/fused (Detection3DArray)
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
    → camera + LiDAR share same sample['timestamp'] → always synchronized

bags/nuscenes_scene0/
    ├── metadata.yaml          ← bag metadata
    └── nuscenes_scene0.db3    ← SQLite database with messages

ros2 bag play bags/nuscenes_scene0
    → /nuscenes/camera/image_raw  @ 2 Hz (nuScenes capture rate)
    → /nuscenes/lidar/pointcloud  @ 2 Hz (synchronized)
```

---

## LiDAR Coordinate Transform

nuScenes LIDAR_TOP is not axis-aligned with the ego vehicle frame.
Transform derived from `calibrated_sensor` quaternion:

```
Quaternion: [0.7077955, -0.006492, 0.010646, -0.7063073]
→ LiDAR +X maps to ego RIGHT
→ LiDAR +Y maps to ego FORWARD

Applied in lidar_detection_node:
  det.bbox.center.position.x = -b.y   ← ego forward
  det.bbox.center.position.y =  b.x   ← ego left

Verified: raw output b.y=25.3 → car 25m ahead confirmed
```

---

## Fusion Node

**BEV Camera Projection:**
```
nuScenes CAM_FRONT: fx=fy=1266.417, cx=816.267, cy=491.507
Camera height: 1.5m above ground

Z = 1.5 × 1266.417 / (v_bottom - 491.507)  ← forward depth
X = (u - 816.267) × Z / 1266.417            ← lateral
BEV = (Z, -X)                                ← (forward, left)
Valid: Z ∈ [1m, 60m]
```

**Matching:**
```
D[i,j] = euclidean(cam_bev[i], lidar_bev[j])
Match if D[i,j] ≤ 8.0m
Assignment: Hungarian greedy (minimize total distance)
Match rate: 40–100% of camera detections per frame
```

**Fusion score:** `0.6 × lidar_score + 0.4 × camera_score`
**Class label:** camera class used when matched (higher semantic confidence)

---

## Performance Analysis

| Condition | FPS | Pre | TRT | Note |
|---|---|---|---|---|
| Standalone C++ (720p) | 203 | 1.1ms | 3.7ms | No ROS2 overhead |
| ROS2 camera_node (720p) | ~195 | 1.1ms | 3.7ms | topic serialize overhead |
| ROS2 camera_node (1600×900) | ~180 | 1.8ms | 4.0ms | nuScenes resolution |
| CUDA-PointPillars FPN3 (43K pts) | ~37-45 | 0.2ms vox | 13-17ms bb | 3-level FPN |
| lidar_detection_node (ROS2) | ~2 | — | ~25-30ms/frame | gated by bag 2Hz |
| fusion_node match rate | — | — | <1ms | 40–100% cam dets matched |

ROS2 overhead is ~5 FPS due to:
- `sensor_msgs/Image` serialization/deserialization per frame
- `cv_bridge::toCvShare` copy (zero-copy not yet implemented)
- DDS middleware latency

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
in the same container (as in `run_ros2_rviz_demo.sh`).

**Power mode required for dual TRT engines**
Running camera_node (INT8) + lidar_detection_node (FP16) simultaneously
requires 25W / 15W mode: `sudo nvpmodel -m 2`. MAXN SUPER mode causes overcurrent.
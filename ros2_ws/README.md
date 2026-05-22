# EdgeDrive Perception — ROS2 Package

ROS2 Humble package for real-time camera detection and sensor fusion
on Jetson Orin Nano Super. Built on the same TensorRT INT8 pipeline
as the standalone C++ deployment.

```
Package  : edgedrive_perception
ROS2     : Humble
Container: edgedrive-ros2:latest (l4t-jetpack:r36.4.0 + ROS2 Humble)
TRT      : 10.3.0 (matches host engines — no rebuild needed)
```

---

## Package Structure

```
ros2_ws/
└── src/
    └── edgedrive_perception/
        ├── package.xml              ← ROS2 dependencies
        ├── CMakeLists.txt           ← build config, reuses deployment/ sources
        ├── include/edgedrive_perception/
        │   ├── camera_node.hpp      ← YOLO26n detection node
        │   ├── lidar_detection_node.hpp ← CUDA-PointPillars ROS2 node
        │   └── fusion_node.hpp      ← camera-LiDAR late fusion
        ├── src/
        │   ├── camera_node.cpp      ← YOLO26n TRT inference
        │   ├── lidar_detection_node.cpp ← CUDA-PointPillars TRT inference
        │   └── fusion_node.cpp      ← BEV matching + MarkerArray visualization
        ├── launch/
        │   └── camera.launch.py     ← configurable launch file
        └── config/
            ├── camera_node.yaml     ← default parameters
            └── edgedrive.rviz       ← RViz2 auto-config (camera + LiDAR + fusion)
```

---

## Nodes

### camera_node

YOLO26n TRT INT8 detection node. Subscribes to a camera image topic,
runs TensorRT inference, publishes detection results.
Barriers are filtered from all outputs (annotated image, Detection2DArray, MarkerArray).

**Subscribes:**
| Topic | Type | Description |
|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` | Input camera frames |

**Publishes:**
| Topic | Type | Description |
|---|---|---|
| `/detections/camera` | `vision_msgs/Detection2DArray` | 2D bounding boxes + class + score |
| `/camera/annotated` | `sensor_msgs/Image` | Annotated frame with boxes + FPS HUD |
| `/camera/bev` | `sensor_msgs/Image` | Bird's Eye View projection (if enabled) |
| `/detections/camera_markers` | `visualization_msgs/MarkerArray` | 3D cylinders in BEV for RViz2 |

**Parameters:**
| Parameter | Default | Description |
|---|---|---|
| `engine_path` | `weights/yolo26n_det_int8_raw.engine` | TRT engine path |
| `score_threshold` | `0.3` | Detection confidence threshold |
| `publish_viz` | `true` | Publish annotated image |
| `publish_bev` | `false` | Publish BEV projection |
| `publish_markers` | `true` | Publish RViz2 MarkerArray (3D BEV cylinders) |
| `camera_height` | `1.2` | Camera height above ground (m) for BEV |

**QoS:** RELIABLE, VOLATILE, depth=10

---

### lidar_detection_node

CUDA-PointPillars FPN3 TRT FP16 detection node. Subscribes to a LiDAR
PointCloud2 topic, runs voxelization → CHW scatter → TRT FPN3 backbone → NMS,
publishes Detection3DArray + MarkerArray (blue cylinders in RViz2).

Detects all 10 nuScenes classes across full 100m × 100m range using 3 FPN levels.

**Subscribes:**
| Topic | Type | Description |
|---|---|---|
| `/nuscenes/lidar/pointcloud` | `sensor_msgs/PointCloud2` | Input LiDAR frames |

**Publishes:**
| Topic | Type | Description |
|---|---|---|
| `/detections/lidar` | `vision_msgs/Detection3DArray` | 3D bounding boxes in ego frame |
| `/detections/lidar_markers` | `visualization_msgs/MarkerArray` | Blue 3D cylinders → RViz2 BEV |

**Parameters:**
| Parameter | Default | Description |
|---|---|---|
| `engine_path` | `cuda-pointpillars/model/pointpillar_fpn3.plan` | TRT FPN3 plan file |
| `score_threshold` | `0.15` | Detection confidence threshold |
| `publish_markers` | `true` | Publish RViz2 MarkerArray |

**Coordinate transform (LiDAR→ego frame):**
```
nuScenes LIDAR_TOP quaternion: [0.7077955, -0.006492, 0.010646, -0.7063073]
→ ego_forward = -LiDAR_Y
→ ego_left    = +LiDAR_X
(derived from calibrated_sensor rotation matrix, verified empirically)
```

**Performance:** ~25-30ms per frame, ~37-45 FPS. FPN3 uses 2 levels in ROS2
(level 0: pedestrian/bicycle, level 1: cars) to stay within memory budget
when running concurrently with camera_node.

**Prerequisites:**
```bash
# Build CUDA-PointPillars and generate TRT engine first
./scripts/setup_cuda_pointpillars.sh
```

**Performance:** ~25-30ms per frame (no jetson_clocks), 2Hz input from nuScenes bag.

**QoS:** RELIABLE, VOLATILE, depth=10 (matches bag replay publisher)

---

### fusion_node

Camera-LiDAR late fusion node. ApproximateTime sync subscribes to both
detection streams and matches camera BEV projections against LiDAR 3D
detections using BEV distance matching.

**Subscribes:**
| Topic | Type | Description |
|---|---|---|
| `/detections/camera` | `vision_msgs/Detection2DArray` | 2D camera detections |
| `/detections/lidar` | `vision_msgs/Detection3DArray` | 3D LiDAR detections |

**Publishes:**
| Topic | Type | Description |
|---|---|---|
| `/detections/fused` | `vision_msgs/Detection3DArray` | Fused + unmatched LiDAR detections |
| `/visualization/fused` | `visualization_msgs/MarkerArray` | Red=fused, Green=cam-only, Blue=lidar-only |

**Parameters:**
| Parameter | Default | Description |
|---|---|---|
| `match_threshold` | `8.0` | BEV distance threshold (meters) |
| `sync_tolerance` | `2.0` | ApproximateTime tolerance (seconds) |
| `camera_fx` | `1266.417` | nuScenes CAM_FRONT intrinsic |
| `camera_fy` | `1266.417` | |
| `camera_cx` | `816.267` | |
| `camera_cy` | `491.507` | |
| `camera_height` | `1.5` | Camera height above ground (m) |

**Camera BEV projection:**
```
Z = camera_height × fy / (v_bottom - cy)   ← forward depth
X = (u - cx) × Z / fx                       ← lateral
BEV = (Z, -X)                               ← (forward, left) in ego frame
```

**Fusion score:** `0.6 × lidar_score + 0.4 × camera_score`
Camera class label used when matched (higher semantic confidence).

**Match rate:** 40–100% of camera detections fused per frame.

**Stats overlay:** Live `CAM: N  LiDAR: N  FUSED: N` text in RViz2 BEV.

See [`docs/sensor_fusion_analysis.md`](../docs/sensor_fusion_analysis.md) for full design details.

---

## Quick Start

### Build

```bash
# Build CUDA-PointPillars .so first
cd ~/EdgeDrive-Perception/cuda-pointpillars/build
export CUDASM=87 && make -j4

# Build Docker image (copies .so + builds all 3 nodes)
cd ~/EdgeDrive-Perception
docker build -t edgedrive-ros2:latest -f docker/Dockerfile.ros2 .
```

Or rebuild nodes only inside container:
```bash
docker compose -f docker/docker-compose.ros2.yml run --rm dev \
    bash -c "cd /workspace/ros2_ws && colcon build --packages-select edgedrive_perception"
```

### Run — full fusion demo with RViz2

```bash
sudo nvpmodel -m 2    # 25W mode — required for dual TRT engines
sudo jetson_clocks
xhost +local:docker
./scripts/run_ros2_rviz_demo.sh
```

Starts TF + bag + camera_node + lidar_detection_node + fusion_node + RViz2
in one container. RViz2 auto-loads:
```
Fixed Frame     : base_link
Green cylinders : /detections/camera_markers  (camera BEV projection)
Blue cylinders  : /detections/lidar_markers   (CUDA-PointPillars 3D)
Red cylinders   : /visualization/fused        (matched detections)
Image panel     : /camera/annotated
```

### Run — live USB camera

```bash
sudo jetson_clocks
docker compose -f docker/docker-compose.ros2.yml run --rm camera-node
```

### Run — nuScenes bag replay

```bash
# Generate one scene
python3 scripts/nuscenes_to_ros2bag.py \
    --dataroot /data/sets/nuscenes \
    --output bags/nuscenes_scene0 --scene-idx 0

# Generate all 10 scenes
python3 scripts/nuscenes_to_ros2bag.py \
    --dataroot /data/sets/nuscenes \
    --output bags/nuscenes_all --all-scenes

# Run headless pipeline
sudo jetson_clocks
./scripts/run_ros2_bag_demo.sh

# Run specific scene or all scenes
BAG=nuscenes_scene3 ./scripts/run_ros2_bag_demo.sh
BAG=nuscenes_all    ./scripts/run_ros2_bag_demo.sh
```

### Run — with launch file

```bash
docker compose -f docker/docker-compose.ros2.yml run --rm camera-launch
```

### Test with synthetic image

```bash
# Terminal 1
docker compose -f docker/docker-compose.ros2.yml run --rm camera-node

# Terminal 2
./scripts/test_ros2_camera.sh
```

---

## Performance

| Mode | FPS | Pre | TRT | Input |
|---|---|---|---|---|
| camera_node (jetson_clocks) | ~180 | 1.8ms | 3.7ms | 1600×900 nuScenes |
| camera_node USB (jetson_clocks) | ~205 | 1.1ms | 3.7ms | 1280×720 |
| lidar_detection_node | ~37-45 | 0.2ms | 13-17ms | 43K pts nuScenes |
| fusion_node match rate | — | — | <1ms | 40–100% cam dets |

FPS is hardware capability — actual detection rate limited by input source
(nuScenes bag publishes at 2 Hz, USB camera at 30 Hz).

---

## Key Design Notes

**Reuses deployment/ sources:**
CMakeLists.txt references `trt_engine.cpp`, `yolo26_decoder.cpp`,
`bev_visualizer.cpp` and `profiler.cpp` directly from `deployment/src/`.
No code duplication — same inference pipeline in both standalone and ROS2.

**QoS matching:**
Subscriber uses RELIABLE + VOLATILE. Publishers must match exactly or
messages are silently dropped (common ROS2 pitfall).

**Topic remap for bag replay:**
```bash
ros2 run edgedrive_perception camera_node --ros-args \
    -r /camera/image_raw:=/nuscenes/camera/image_raw
```

**Container vs host binary:**
The compiled binaries live inside the container at build time.
Mounting `-v ~/EdgeDrive-Perception:/workspace` overwrites them.
Use compose services (camera-node, lidar-node) which mount
only specific subdirectories.

**Power mode for fusion demo:**
Running camera_node (INT8) + lidar_detection_node (FP16) requires 15W or 25W:
```bash
sudo nvpmodel -m 2
```
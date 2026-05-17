# EdgeDrive Perception — ROS2 Package

ROS2 Humble package for real-time camera detection and sensor fusion
on Jetson Orin Nano Super. Built on the same TensorRT INT8 pipeline
as the standalone C++ deployment.

```
Package  : edgedrive_perception
ROS2     : Humble
Container: edgedrive-ros2:latest (l4t-jetpack:r36.4.0 + ROS2 Humble)
TRT      : 10.3.0 (matches host engines — no rebuild needed)
Status   : camera_node ✅ | fusion_node ⬜
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
        │   └── fusion_node.hpp      ← camera-LiDAR fusion (⬜ planned)
        ├── src/
        │   └── camera_node.cpp      ← full implementation
        ├── launch/
        │   └── camera.launch.py     ← configurable launch file
        └── config/
            └── camera_node.yaml     ← default parameters
```

---

## Nodes

### camera_node

YOLO26n TRT INT8 detection node. Subscribes to a camera image topic,
runs TensorRT inference, publishes detection results.

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

**Parameters:**
| Parameter | Default | Description |
|---|---|---|
| `engine_path` | `weights/yolo26n_det_int8_raw.engine` | TRT engine path |
| `score_threshold` | `0.3` | Detection confidence threshold |
| `publish_viz` | `true` | Publish annotated image |
| `publish_bev` | `false` | Publish BEV projection |
| `camera_height` | `1.2` | Camera height above ground (m) for BEV |

**QoS:** RELIABLE, VOLATILE, depth=10

---

### fusion_node (planned — June 2026)

Camera-LiDAR late fusion node using ApproximateTime synchronization.

**Subscribes:**
| Topic | Type | Description |
|---|---|---|
| `/camera/image_raw` | `sensor_msgs/Image` | Camera frames |
| `/lidar/pointcloud` | `sensor_msgs/PointCloud2` | LiDAR point cloud |

**Publishes:**
| Topic | Type | Description |
|---|---|---|
| `/detections/fused` | `vision_msgs/Detection3DArray` | Fused 3D detections |
| `/visualization/fused` | `visualization_msgs/MarkerArray` | RViz2 markers |

Requires CUDA-PointPillars on Jetson for LiDAR branch.
See [`docs/sensor_fusion_analysis.md`](../docs/sensor_fusion_analysis.md).

---

## Quick Start

### Build

```bash
# Inside edgedrive-ros2 container
docker compose -f docker/docker-compose.ros2.yml run --rm dev \
    bash -c "cd /workspace/ros2_ws && colcon build --packages-select edgedrive_perception"
```

Or rebuild the full image:
```bash
docker build -t edgedrive-ros2:latest -f docker/Dockerfile.ros2 .
```

### Run — live USB camera

```bash
sudo jetson_clocks
docker compose -f docker/docker-compose.ros2.yml run --rm camera-node
```

### Run — nuScenes bag replay

```bash
# Generate bag first (one time)
python3 scripts/nuscenes_to_ros2bag.py \
    --dataroot /data/sets/nuscenes \
    --output bags/nuscenes_scene0 \
    --scene-idx 0

# Run full pipeline
sudo jetson_clocks
./scripts/run_ros2_bag_demo.sh
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
| nuScenes bag (jetson_clocks) | ~180 | 1.8ms | 3.7ms | 1600×900 |
| USB camera (jetson_clocks) | ~205 | 1.1ms | 3.7ms | 1280×720 |

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
The compiled binary lives inside the container at build time.
Mounting `-v ~/EdgeDrive-Perception:/workspace` overwrites it.
Use compose services (camera-node, camera-node-bev) which mount
only specific subdirectories.

# EdgeDrive Perception — Docker

Two containers, same base image (`l4t-jetpack:r36.4.0`), different purposes.

| Container | Image | Size | TRT | Purpose |
|---|---|---|---|---|
| `edgedrive` | `Dockerfile` | ~10GB | 10.3.0 | Production C++ runtime |
| `edgedrive-ros2` | `Dockerfile.ros2` | ~14GB | 10.3.0 | ROS2 + CUDA-PointPillars + fusion |

Both use `l4t-jetpack:r36.4.0` → same TRT 10.3.0 → same engines work in both.
No engine rebuild needed when switching between containers.

---

## Container 1 — Production C++ Runtime

```
Dockerfile
docker-compose.yml

Base   : nvcr.io/nvidia/l4t-jetpack:r36.4.0
Adds   : GStreamer runtime plugins
Binary : edgedrive (compiled inside, multi-stage build)
Size   : ~10GB
Status : ✅ tested — 203 FPS / 8.03W on Jetson Orin Nano Super
```

### Build

```bash
cd ~/EdgeDrive-Perception
docker build -t edgedrive:latest -f docker/Dockerfile .
```

```
First build  : ~15-20 min (downloads base + installs packages + compiles C++)
Cached rebuild: ~16s      (only recompiles changed sources)
```

### Run

```bash
sudo jetson_clocks        # lock GPU to 1017 MHz (required for full speed)
xhost +local:docker       # allow X11 display

# USB camera — detection only
docker compose -f docker/docker-compose.yml run --rm camera

# USB camera — detection + BEV split-screen
docker compose -f docker/docker-compose.yml run --rm camera-bev

# Pre-recorded video — detection only
VIDEO=tokyo.mp4 docker compose -f docker/docker-compose.yml run --rm video

# Pre-recorded video — detection + BEV
VIDEO=tokyo.mp4 docker compose -f docker/docker-compose.yml run --rm video-bev

# Save annotated output (headless)
VIDEO=tokyo.mp4 docker compose -f docker/docker-compose.yml run --rm save
# → output/demo_annotated.mp4

# Save BEV side-by-side output (headless)
VIDEO=tokyo.mp4 docker compose -f docker/docker-compose.yml run --rm save-bev
# → output/demo_bev.mp4

# Benchmark (headless, jetson_clocks required)
docker compose -f docker/docker-compose.yml run --rm benchmark
# Expected: ~193 FPS / 5.1ms / ~12.2W
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VIDEO` | `nuscenes_clip.mp4` | Video filename in `test_videos/` |
| `THRESH` | `0.3` | Detection score threshold |
| `DURATION` | `60` | Benchmark duration (seconds) |

### Volume Mounts

| Host path | Container path | Access |
|---|---|---|
| `weights/` | `/workspace/weights` | read-only |
| `test_images/` | `/workspace/test_images` | read-only |
| `test_videos/` | `/workspace/test_videos` | read-only |
| `calibration.yaml` | `/workspace/calibration.yaml` | read-only |
| `output/` | `/workspace/output` | read-write |

### Multi-Stage Build

```
Stage 1 — builder (l4t-jetpack:r36.4.0):
  Installs: cmake, ninja, g++, GStreamer dev headers
  Compiles: edgedrive C++ binary from source
  Note: -Wl,--allow-shlib-undefined skips BSP libs missing at build time
        (libnvdla_compiler.so, libnvcudla.so — injected at runtime by
        --runtime nvidia from host JetPack)

Stage 2 — runtime (l4t-jetpack:r36.4.0):
  Installs: GStreamer runtime plugins only
  Copies:   edgedrive binary from builder
  Result:   ~10GB (same base, binary + plugins only added on top)
```

---

## Container 2 — ROS2 Development

```
Dockerfile.ros2
docker-compose.ros2.yml
ros2_entrypoint.sh

Base   : nvcr.io/nvidia/l4t-jetpack:r36.4.0 (same as production)
Adds   : ROS2 Humble Desktop, colcon, cv_bridge, vision_msgs,
         visualization_msgs, rosbag2, tf2_ros
         libpointpillar_core.so (CUDA-PointPillars, pre-built on host)
Binary : camera_node, lidar_detection_node, fusion_node (colcon at build time)
Size   : ~14GB
Status : camera_node ~180 FPS | lidar_detection_node ~37-45 FPS | fusion_node
```

### Why same base as production?

```
dusty-nv/jetson-containers alternative:
  ❌ Ships TRT 10.4.0 — mismatches our TRT 10.3.0 engines
  ❌ PyTorch CUDA broken (torch.cuda.is_available() = False)
  ❌ pip conflicts — can't install packages cleanly
  ❌ 15-20GB image

Our custom Dockerfile.ros2:
  ✅ Same TRT 10.3.0 as host — engines work immediately
  ✅ CUDA fully functional
  ✅ Clean package install
  ✅ ~14GB
```

### Build

```bash
# Build CUDA-PointPillars .so on host first
cd ~/EdgeDrive-Perception/cuda-pointpillars/build
export CUDASM=87
make -j4

# Build Docker image (copies .so + builds all 3 ROS2 nodes)
cd ~/EdgeDrive-Perception
docker build -t edgedrive-ros2:latest -f docker/Dockerfile.ros2 .
```

```
First build  : ~30-40 min (ROS2 Humble + deps + colcon build)
Cached rebuild: ~30s      (only recompiles when source changes)
  - Package manifests (package.xml, CMakeLists.txt) cached separately
  - Source changes only recompile the C++ node (~25s)
```

> **Note:** `libpointpillar_core.so` is copied from the host at build time.
> If you change `lidar-backbone.cu` or `pointpillar.cpp`, run `make -j4`
> in `cuda-pointpillars/build` before rebuilding Docker.

### Verify

```bash
# Check TRT version matches host
docker run --rm --runtime nvidia edgedrive-ros2:latest \
    bash -c "python3 -c 'import tensorrt; print(tensorrt.__version__)'"
# Expected: 10.3.x
```

### Run — camera_node (live USB camera)

```bash
sudo jetson_clocks
xhost +local:docker

# Detection only
docker compose -f docker/docker-compose.ros2.yml run --rm camera-node

# Detection + BEV
docker compose -f docker/docker-compose.ros2.yml run --rm camera-node-bev
```

### Run — full fusion demo (camera + LiDAR + fusion + RViz2)

```bash
# Generate bag first (one time)
python3 scripts/nuscenes_to_ros2bag.py \
    --dataroot /data/sets/nuscenes \
    --output bags/nuscenes_scene0 --scene-idx 0

sudo nvpmodel -m 2    # 25W mode — required for dual TRT engines
sudo jetson_clocks
xhost +local:docker
./scripts/run_ros2_rviz_demo.sh
```

Starts in a single container: TF publisher + bag replay + camera_node +
lidar_detection_node + fusion_node + RViz2.

RViz2 auto-loads `config/edgedrive.rviz`:
```
Fixed Frame     : base_link
Green cylinders : /detections/camera_markers   (camera BEV projection)
Blue cylinders  : /detections/lidar_markers    (CUDA-PointPillars 3D)
Red cylinders   : /visualization/fused         (matched detections)
Image panel     : /camera/annotated            (annotated camera feed)
Stats overlay   : live CAM / LiDAR / FUSED count per frame
```

### Run — headless (no RViz2)

```bash
# Headless pipeline (bag → TRT → /detections/camera)
sudo jetson_clocks
./scripts/run_ros2_bag_demo.sh
```

### Topics published by camera_node

| Topic | Type | Description |
|---|---|---|
| `/detections/camera` | `vision_msgs/Detection2DArray` | 2D boxes + class + score (barriers excluded) |
| `/camera/annotated` | `sensor_msgs/Image` | Annotated frame + FPS HUD (barriers excluded) |
| `/camera/bev` | `sensor_msgs/Image` | BEV projection (if enabled) |
| `/detections/camera_markers` | `visualization_msgs/MarkerArray` | Green 3D cylinders → RViz2 |

### Topics published by lidar_detection_node

| Topic | Type | Description |
|---|---|---|
| `/detections/lidar` | `vision_msgs/Detection3DArray` | 3D boxes in ego frame, 10 classes |
| `/detections/lidar_markers` | `visualization_msgs/MarkerArray` | Blue 3D cylinders → RViz2 |

### Topics published by fusion_node

| Topic | Type | Description |
|---|---|---|
| `/detections/fused` | `vision_msgs/Detection3DArray` | Matched + unmatched LiDAR detections |
| `/visualization/fused` | `visualization_msgs/MarkerArray` | Red=fused, Green=cam-only, Blue=lidar-only |

### QoS — important

ROS2 drops messages silently on QoS mismatch. All topics use:
```
Reliability : RELIABLE
Durability  : VOLATILE
History     : KEEP_LAST (depth 10)
```

### Key design notes

**Binary lives inside the container:**
The node binaries are compiled at `docker build` time and live at
`/workspace/ros2_ws/install/`. Mounting `-v ~/EdgeDrive-Perception:/workspace`
overwrites this — use compose services which mount only specific subdirectories.

**DDS discovery between containers:**
Even with `--network host`, DDS multicast sometimes fails between separate
containers. Solution: run bag replay and nodes in the same container
(as in `run_ros2_rviz_demo.sh`).

**colcon build output stays in container:**
`ros2_ws/build/`, `ros2_ws/install/`, `ros2_ws/log/` are gitignored.
The compiled binaries are baked into the image — no host build artifacts needed.

### Compose services

| Service | Description |
|---|---|
| `camera-node` | camera_node on USB camera (/dev/video0) |
| `camera-node-bev` | camera_node + BEV on USB camera |
| `camera-node-bag` | camera_node remapped to nuScenes bag topics |
| `camera-launch` | camera_node via launch file |
| `lidar-node` | lidar_detection_node (CUDA-PointPillars FPN3, 10 classes, ~37-45 FPS) |
| `bag-replay` | ros2 bag play (loop) |
| `rviz` | RViz2 only |
| `dev` | Interactive shell for development |

---

## First-Time Setup (both containers)

### 1. Enable NVIDIA Container Runtime

```bash
# Verify
docker run --rm --runtime nvidia ubuntu:22.04 echo "ok"

# If it fails:
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl restart docker
```

### 2. Allow X11 display

```bash
xhost +local:docker
# Add to ~/.bashrc to persist
echo "xhost +local:docker" >> ~/.bashrc
```

### 3. Lock GPU clocks

```bash
sudo jetson_clocks
```

> Must run on **host** before any container. Sets hardware registers —
> container inherits host clock state. Without it: ~50 FPS instead of 200+.

---

## Troubleshooting

**`Unknown runtime specified nvidia`**
```bash
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl restart docker
```

**`cannot connect to X server`**
```bash
xhost +local:docker
```

**Low FPS (~50 instead of 200+)**
```bash
sudo jetson_clocks  # run on host before starting container
```

**`NvMapMemAllocInternalTagged error` (CUDA OOM in ROS2)**
Add NVMAP devices to compose service:
```yaml
devices:
  - /dev/nvmap:/dev/nvmap
  - /dev/nvhost-ctrl-gpu:/dev/nvhost-ctrl-gpu
  - /dev/nvhost-gpu:/dev/nvhost-gpu
```

**Overcurrent / system crash with dual TRT engines**
```bash
sudo nvpmodel -m 2   # 25W mode before starting the fusion demo
```
Running camera_node (INT8) + lidar_detection_node (FP16) simultaneously
in MAXN SUPER mode causes overcurrent. 15W or 25W mode is stable.

**`TRT engine plan file not compatible`**
Engine was built with different TRT version than container.
Both containers use TRT 10.3.0 — engines built on host (JetPack R36.4.7)
are compatible with both containers.

**`unable to start pipeline` (NVDEC warning)**
Harmless — falls back to software decode automatically.

**`not found: /opt/ros/humble/install/local_setup.bash`**
Harmless warning from entrypoint. No impact on functionality.

**RViz2 `frame [base_link] does not exist`**
TF publisher not running. Use `run_ros2_rviz_demo.sh` which starts
the static TF publisher automatically.

**Engine not found**
Engines are not baked into the image — build on host first:
```bash
./scripts/build_engine.sh int8                    # camera engine
./scripts/setup_cuda_pointpillars.sh              # LiDAR engine
```
# EdgeDrive Perception — Docker

Production runtime container for Jetson Orin Nano Super.
Isolates the C++ TRT deployment stack from the host JetPack environment.

```
Base image : nvcr.io/nvidia/l4t-jetpack:r36.4.0
CUDA       : 12.6
TensorRT   : 10.x
OpenCV     : 4.8.0
Image size : ~10GB (l4t-jetpack includes full CUDA + TRT + cuDNN + OpenCV)
Binary     : edgedrive (compiled inside container, multi-stage build)
Status     : ✅ tested and working on Jetson Orin Nano Super 8GB
```

> **Note:** This container is for the production C++ pipeline only.
> The ROS2 development environment uses a separate image.

---

## Build Times

```
First build (no cache) : ~15-20 min  (downloads base image + installs packages)
Cached rebuild         : ~16s        (only recompiles C++ when source changes)
Docker layer caching means rebuilding after source changes is very fast.
```

---

## Requirements

- Jetson Orin Nano Super with JetPack R36.4.0 or newer
- Docker 20.10+
- NVIDIA Container Runtime (included with JetPack)
- ~15GB free disk space (10GB image + build cache)

---

## First-Time Setup

### 1. Verify NVIDIA Container Runtime

```bash
docker run --rm --runtime nvidia \
    ubuntu:22.04 echo "nvidia runtime ok"
```

If it fails:
```bash
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl restart docker
```

### 2. Allow X11 display (for imshow window)

```bash
xhost +local:docker
```

Add to `~/.bashrc` to persist:
```bash
echo "xhost +local:docker" >> ~/.bashrc
```

### 3. Lock GPU clocks (required for full performance)

```bash
sudo jetson_clocks
```

> Must be run on the **host** before starting any container.
> `jetson_clocks` sets hardware clock registers — the container inherits
> the host clock state. Without it, GPU idles at ~300 MHz → ~50 FPS
> instead of 1017 MHz → 200+ FPS.

### 4. Build the image

```bash
# From repo root — must be built on Jetson (ARM64 only)
cd ~/EdgeDrive-Perception
docker build -t edgedrive:latest -f docker/Dockerfile .
```

### 5. Verify

```bash
docker images edgedrive
# edgedrive   latest   xxxx   ~10GB
```

---

## Running

All commands from repo root:

```bash
cd ~/EdgeDrive-Perception
```

### USB camera — detection only

```bash
sudo jetson_clocks
docker compose -f docker/docker-compose.yml run --rm camera
```

### USB camera — detection + BEV

```bash
sudo jetson_clocks
docker compose -f docker/docker-compose.yml run --rm camera-bev
```

### Pre-recorded video — detection only

```bash
VIDEO=tokyo.mp4 docker compose -f docker/docker-compose.yml run --rm video
```

### Pre-recorded video — detection + BEV

```bash
VIDEO=tokyo.mp4 docker compose -f docker/docker-compose.yml run --rm video-bev
```

### Save annotated video (headless)

```bash
VIDEO=tokyo.mp4 docker compose -f docker/docker-compose.yml run --rm save
# Output: output/demo_annotated.mp4
```

### Save BEV side-by-side video (headless)

```bash
VIDEO=tokyo.mp4 docker compose -f docker/docker-compose.yml run --rm save-bev
# Output: output/demo_bev.mp4
```

### Benchmark

```bash
sudo jetson_clocks
docker compose -f docker/docker-compose.yml run --rm benchmark
# Expected: ~193 FPS, 5.1ms, ~12.2W
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VIDEO` | `nuscenes_clip.mp4` | Video filename in `test_videos/` |
| `THRESH` | `0.3` | Detection score threshold |
| `DURATION` | `60` | Benchmark duration (seconds) |

Example:
```bash
VIDEO=tokyo.mp4 THRESH=0.5 docker compose -f docker/docker-compose.yml run --rm video-bev
```

---

## Volume Mounts

| Host path | Container path | Access |
|---|---|---|
| `weights/` | `/workspace/weights` | read-only |
| `test_images/` | `/workspace/test_images` | read-only |
| `test_videos/` | `/workspace/test_videos` | read-only |
| `calibration.yaml` | `/workspace/calibration.yaml` | read-only |
| `output/` | `/workspace/output` | read-write |

Engines and videos are **not baked into the image** — mounted at runtime.
Swap engines or videos without rebuilding the image.

---

## Multi-Stage Build

```
Stage 1 — builder (l4t-jetpack:r36.4.0):
  Installs: cmake, ninja, g++, GStreamer dev headers
  Compiles: edgedrive C++ binary from source
  Note: -Wl,--allow-shlib-undefined skips missing Jetson BSP libs
        (libnvdla_compiler.so, libnvcudla.so) at link time — they
        are injected at runtime by --runtime nvidia from the host

Stage 2 — runtime (l4t-jetpack:r36.4.0):
  Installs: GStreamer runtime plugins only
  Copies:   edgedrive binary from builder stage
  Result:   same base image, binary + plugins only added on top
```

### Why 10GB?

`l4t-jetpack:r36.4.0` includes the full JetPack SDK (CUDA, TRT, cuDNN,
OpenCV, all dev tools). Using a smaller runtime base (e.g. `ubuntu:22.04`)
would require manually copying 20+ shared libraries — fragile and
maintenance-heavy. 10GB is the practical tradeoff for a reliable,
self-contained container.

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
# Run on host before starting container
sudo jetson_clocks
```

**`unable to start pipeline` (NVDEC warning)**
Harmless — automatically falls back to software decode.

**Engine not found**
Engines are not in the image — build on host first:
```bash
./scripts/build_engine.sh int8
```

**Container exits immediately**
Check the engine path is correct and weights/ is mounted:
```bash
docker compose -f docker/docker-compose.yml run --rm camera \
    --engine weights/yolo26n_det_int8_raw.engine
```

---

## ROS2 Development Container

A second container adds ROS2 Humble on top of the same `l4t-jetpack:r36.4.0` base.
Same TRT 10.3.0 — no engine rebuild needed.

```
Dockerfile.ros2       ← ROS2 dev container
docker-compose.ros2.yml
ros2_entrypoint.sh
```

### Build ROS2 image

```bash
cd ~/EdgeDrive-Perception
docker build -t edgedrive-ros2:latest -f docker/Dockerfile.ros2 .
```

### Run camera_node via ROS2

```bash
sudo jetson_clocks
xhost +local:docker

# Camera detection node
docker compose -f docker/docker-compose.ros2.yml run --rm camera-node

# Camera node + BEV visualization
docker compose -f docker/docker-compose.ros2.yml run --rm camera-node-bev

# Interactive dev shell
docker compose -f docker/docker-compose.ros2.yml run --rm dev
```

### Verify topics

```bash
# In a second terminal while camera-node is running
docker compose -f docker/docker-compose.ros2.yml run --rm dev \
    ros2 topic list
# /detections/camera
# /camera/annotated
# /camera/bev  (if publish_bev=true)
```

### Two containers summary

| Container | Image | Size | TRT | Purpose |
|---|---|---|---|---|
| `edgedrive` | `Dockerfile` | ~10GB | 10.3.0 | Production C++ runtime |
| `edgedrive-ros2` | `Dockerfile.ros2` | ~14GB | 10.3.0 | ROS2 development + fusion |

Both use `l4t-jetpack:r36.4.0` as base → same TRT version → same engines work in both.

### Delete dusty-nv image (no longer needed)

```bash
docker rmi dustynv/ros:humble-desktop-l4t-r36.4.0
```
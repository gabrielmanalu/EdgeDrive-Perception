# EdgeDrive Perception — Docker

Production runtime container for Jetson Orin Nano Super.
Isolates the C++ TRT deployment stack from the host JetPack environment.

```
Base image : nvcr.io/nvidia/l4t-tensorrt:r36.3.0-runtime
CUDA       : 12.6
TensorRT   : 10.x
OpenCV     : 4.8.0 (with GStreamer)
Binary     : edgedrive (compiled inside container, multi-stage build)
```

> **Note:** This container is for the production C++ pipeline only.
> The ROS2 development environment uses a separate image based on
> dusty-nv/jetson-containers (see roadmap for June 2026).

---

## Requirements

- Jetson Orin Nano Super with JetPack R36.3.0 or newer
- Docker 20.10+
- NVIDIA Container Runtime (included with JetPack)

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

Add to `~/.bashrc` to persist across reboots:
```bash
echo "xhost +local:docker" >> ~/.bashrc
```

### 3. Lock GPU clocks (required for full performance)

```bash
sudo jetson_clocks
```

> This must be run on the **host** before starting any container.
> `jetson_clocks` sets hardware clock registers — the container inherits
> the host clock state. Without it, GPU idles at ~300 MHz → ~50 FPS
> instead of 1017 MHz → 200+ FPS.

### 4. Build the image

```bash
# From repo root — must be built on Jetson (ARM64, TRT engines are device-specific)
cd ~/EdgeDrive-Perception
docker build -t edgedrive:latest -f docker/Dockerfile .
```

Expected build time: ~10-15 minutes (pulls base image + compiles C++ from source).

---

## Running

All commands run from the repo root:

```bash
cd ~/EdgeDrive-Perception
```

### USB camera — detection only

```bash
docker compose -f docker/docker-compose.yml run --rm camera
```

### USB camera — detection + BEV

```bash
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
# Requires jetson_clocks on host first
sudo jetson_clocks
docker compose -f docker/docker-compose.yml run --rm benchmark
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VIDEO` | `nuscenes_clip.mp4` | Video file in `test_videos/` |
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

Engines and videos are **not baked into the image** — they are mounted
at runtime. Swap engines without rebuilding.

---

## Multi-stage Build

```
Stage 1 — builder (nvcr.io/nvidia/l4t-tensorrt:r36.3.0-devel):
  Installs cmake, g++, OpenCV dev, GStreamer dev
  Compiles C++ pipeline from source
  Output: /build/build/edgedrive

Stage 2 — runtime (nvcr.io/nvidia/l4t-tensorrt:r36.3.0-runtime):
  Installs OpenCV runtime, GStreamer runtime only
  Copies edgedrive binary from builder
  Final image: ~2-3 GB (vs ~8 GB builder)
```

---

## Troubleshooting

**`docker: Error response from daemon: Unknown runtime specified nvidia`**
```bash
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
sudo systemctl restart docker
```

**`cannot connect to X server` (imshow fails)**
```bash
xhost +local:docker
# Then re-run the container
```

**Low FPS (~50 instead of 200+)**
```bash
# jetson_clocks not set — run on host before starting container
sudo jetson_clocks
```

**`unable to start pipeline` (NVDEC warning)**
Harmless — falls back to software decode automatically.
NVDEC requires `/dev/nvhost-*` devices. Add to docker-compose if needed:
```yaml
devices:
  - /dev/nvhost-ctrl:/dev/nvhost-ctrl
  - /dev/nvhost-vic:/dev/nvhost-vic
```

**Engine not found**
Engines are not in the image. Mount them or build on the host first:
```bash
./scripts/build_engine.sh int8
```
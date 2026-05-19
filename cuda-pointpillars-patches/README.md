# CUDA-PointPillars Patches — Jetson Orin Nano Super + nuScenes

Minimal patches to run [NVIDIA-AI-IOT/CUDA-PointPillars](https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars)
on **Jetson Orin Nano Super (SM87, JetPack R36.4.7)** with our **nuScenes-trained mmdetection3d model**.

The original repo targets KITTI + JetPack 5.x (TRT 8.x, CUDA 11.4, C++14).
These patches bring it to JetPack 6.x (TRT 10.3, CUDA 12.6, C++17).

---

## What Changed

| File | Change |
|---|---|
| `CMakeLists.txt` | `-std=c++14` → `-std=c++17` (CUDA 12.6 requires C++17) |
| `tensorrt.cpp` | Full TRT 10.x API port (binding API removed in TRT 10) |
| `lidar-backbone.cu` | CHW scatter kernel + backbone-only engine (no PPScatterPlugin) |
| `main.cpp` | nuScenes voxelization + postprocess params |
| `lidar-postprocess.hpp` | nuScenes 10 classes, 8 anchors, bbox code size 9 |

---

## Architecture Change

**Original KITTI pipeline:**
```
points → [Voxelization C++] → [Single TRT engine: VFE + PPScatterPlugin + Backbone + Head]
                                        ↑ 6 bindings: voxels, coords, params, cls, box, dir
```

**Our nuScenes pipeline:**
```
points → [Voxelization C++] → [CHW Scatter CUDA kernel] → [TRT backbone engine] → [Postprocess C++]
                                         ↑ produces                ↑ 4 bindings:
                               [1, 64, 400, 400] float32        pseudo_image → cls, box, dir
```

Why the split? Our model was exported from mmdetection3d as backbone+neck+head only
(SECOND + FPN + Anchor3DHead). The VFE and scatter steps run in C++ using the
existing CUDA kernels — no PPScatterPlugin needed in the TRT engine.

---

## Performance (Jetson Orin Nano Super)

```
                    KITTI data (~19K pts)    nuScenes data (~43K pts)
Voxelization      :  0.2ms                   0.2ms
Backbone+Head     : 12-15ms                  13-17ms
Decoder+NMS       :  6-11ms                  8-12ms
Total (warm)      : 20-25ms  (~45 FPS)       25-30ms  (~37 FPS)
Total (1st frame) : ~140ms   (GPU warm-up)   ~140ms   (GPU warm-up)

Detections        : 6-58 per frame           140-191 per frame
                    (sparse suburban)        (dense urban nuScenes)
```

---

## Setup

### One-command setup (recommended)

```bash
# Export ONNX from mmdetection3d first (see fusion/train_pointpillars.py)
# Then copy to weights/:
cp pointpillars_nuscenes_backbone.onnx ~/EdgeDrive-Perception/weights/

# Run setup script — clones, patches, builds, generates engine, tests
cd ~/EdgeDrive-Perception
./scripts/setup_cuda_pointpillars.sh
```
---

### Manual setup
### 1. Prerequisites
```bash
# Jetson Orin Nano Super with JetPack R36.4.7
# CUDA 12.6, TRT 10.3.0, SM87
sudo jetson_clocks  # lock clocks before benchmarking
```

### 2. Clone and patch

```bash
cd ~/EdgeDrive-Perception

git clone https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars.git cuda-pointpillars
cd cuda-pointpillars

# Apply C++17 fix
sed -i 's/-std=c++14/-std=c++17/g' CMakeLists.txt

# Apply our patches
cp ../cuda-pointpillars-patches/tensorrt.cpp         src/common/tensorrt.cpp
cp ../cuda-pointpillars-patches/lidar-backbone.cu     src/pointpillar/lidar-backbone.cu
cp ../cuda-pointpillars-patches/main.cpp              src/main.cpp
cp ../cuda-pointpillars-patches/lidar-postprocess.hpp src/pointpillar/lidar-postprocess.hpp
```

### 3. Build TRT engine from ONNX

```bash
# Export backbone ONNX from mmdetection3d (see fusion/train_pointpillars.py)
# Then build TRT FP16 engine:

export TensorRT_Bin=/usr/src/tensorrt/bin/

$TensorRT_Bin/trtexec \
    --onnx=model/pointpillars_nuscenes_backbone.onnx \
    --fp16 \
    --plugins=build/libpointpillar_core.so \
    --saveEngine=model/pointpillar.plan \
    --verbose 2>&1 | tail -5

# Expected: &&&& PASSED, GPU latency ~15ms (no jetson_clocks)
```

### 4. Compile and run

```bash
mkdir -p build && cd build

export CUDASM=87
export TensorRT_Lib=/usr/lib/aarch64-linux-gnu/
export TensorRT_Inc=/usr/include/aarch64-linux-gnu/
export TensorRT_Bin=/usr/src/tensorrt/bin/
export CUDA_Lib=/usr/local/cuda-12.6/targets/aarch64-linux/lib/
export CUDA_Inc=/usr/local/cuda-12.6/targets/aarch64-linux/include/
export CUDA_Bin=/usr/local/cuda-12.6/bin/
export CUDA_HOME=/usr/local/cuda-12.6/
export CUDNN_Lib=/usr/lib/aarch64-linux-gnu/

cmake .. && make -j4

# Run on nuScenes LiDAR data
cp /data/sets/nuscenes/samples/LIDAR_TOP/<scene>.pcd.bin ../data/
./pointpillar ../data/ ../data/ --timer
```

---

## nuScenes Model Parameters

```
ONNX model      : pointpillars_nuscenes_backbone.onnx (18.9MB)
Config          : pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py (mmdetection3d)
Checkpoint      : pointpillars_nuscenes.pth

Architecture    : HardVFE → PointPillarsScatter → SECOND → FPN → Anchor3DHead
Exported layers : SECOND + FPN + Anchor3DHead only (backbone+neck+head)

Point cloud range : [-50, -50, -5, 50, 50, 3] meters
Voxel size        : [0.25, 0.25, 8.0] meters
Grid size         : 400 × 400
Pillar features   : 64 channels
Point features    : 4 (x, y, z, intensity) — ring channel dropped
Max voxels        : 30,000
Max points/voxel  : 32

Classes (10): car, truck, bus, trailer, construction_vehicle,
              pedestrian, motorcycle, bicycle, traffic_cone, barrier

Anchors (8 per location):
  car-ish      [2.60×0.87×1.0] rot=[0°, 90°]
  pedestrian   [1.73×0.58×1.0] rot=[0°, 90°]
  bicycle      [1.00×1.00×1.0] rot=[0°, 90°]
  cone/barrier [0.40×0.40×1.0] rot=[0°, 90°]

Score threshold : 0.05
NMS threshold   : 0.20
Max detections  : 500
BBox code size  : 9 (DeltaXYZWLHRBBoxCoder)
```

---

## Known Issues

**`CUDASM` env var not persisted across reboots:**
`export CUDASM=87` must be set in every new terminal before `cmake`.
Add to `~/.bashrc` to persist:
```bash
echo "export CUDASM=87" >> ~/.bashrc
```

**High detection count (~150-179) on nuScenes:**
The model is running correctly. nuScenes scenes are dense urban environments
with many cars, pedestrians, barriers and cones. Scores above 0.05 threshold.
Raise `score_thresh` to 0.3+ to reduce to most confident detections only.

**First frame slower (~50ms) than subsequent (~20ms):**
GPU warm-up effect — TRT kernel compilation on first inference.
Normal behaviour, not a bug.
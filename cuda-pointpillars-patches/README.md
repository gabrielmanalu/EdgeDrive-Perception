# CUDA-PointPillars Patches — Jetson Orin Nano Super + nuScenes

Minimal patches to run [NVIDIA-AI-IOT/CUDA-PointPillars](https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars)
on **Jetson Orin Nano Super (SM87, JetPack R36.4.7)** with our **nuScenes-trained mmdetection3d FPN3 model**.

The original repo targets KITTI + JetPack 5.x (TRT 8.x, CUDA 11.4, C++14).
These patches bring it to JetPack 6.x (TRT 10.3, CUDA 12.6, C++17)
and add full 3-level FPN support for nuScenes 10-class detection.

---

## What Changed

| File | Change |
|---|---|
| `CMakeLists.txt` | `-std=c++14` → `-std=c++17` (CUDA 12.6 requires C++17) |
| `tensorrt.cpp` | Full TRT 10.x API port (binding API removed in TRT 10) |
| `lidar-backbone.hpp` | Added multi-level virtual accessors (`cls1/box1/dir1`, `cls2/box2/dir2`, `is_fpn3()`) |
| `lidar-backbone.cu` | CHW scatter kernel + FPN3 backbone engine (10 bindings, 3 output levels) |
| `pointpillar.hpp` | Added `max_levels` to `CoreParameter` (1=level0, 2=+cars, 3=full FPN3; default 3) |
| `pointpillar.cpp` | Runs postprocess on all FPN levels, merges results; `make_level_param()` scales anchors per level |
| `main.cpp` | nuScenes voxelization + postprocess params; FPN3 engine path |
| `lidar-postprocess.hpp` | nuScenes 10 classes, 8 anchors, bbox code size 9, nuScenes score/NMS thresholds |
| `lidar-postprocess.cu` | NCHW decode fix; NMS safety cap; zero-detection guard; `nms_pre=1000` cap |

---

## Architecture Change

**Original KITTI pipeline:**
```
points → [Voxelization C++] → [Single TRT engine: VFE + PPScatterPlugin + Backbone + Head]
                                        ↑ 6 bindings: voxels, coords, params, cls, box, dir
```

**Our nuScenes FPN3 pipeline:**
```
points → [Voxelization C++] → [CHW Scatter CUDA kernel] → [TRT FPN3 backbone] → [3× Postprocess C++]
                                        ↑ produces             ↑ 10 bindings:        ↑ results merged
                              [1, 64, 400, 400] float32   pseudo_image →
                                                          cls_l0/box_l0/dir_l0 (200×200)
                                                          cls_l1/box_l1/dir_l1 (100×100)
                                                          cls_l2/box_l2/dir_l2 ( 50×50)
```

Why the split? Our model was exported from mmdetection3d as backbone+neck+head only
(SECOND + FPN + Anchor3DHead). The VFE and scatter steps run in C++ using the
existing CUDA kernels — no PPScatterPlugin needed in the TRT engine.

Why FPN3? The original single-level export (feat[0] only) missed all cars and large
objects — they are detected at level 1 (100×100, ×2 anchor scale) and level 2
(50×50, ×4 anchor scale). All 3 levels are required for complete nuScenes detection.

---

## FPN Level Mapping

| Level | Grid | Anchor scale | Primary detections |
|---|---|---|---|
| 0 | 200×200 | ×1 | pedestrian, bicycle, traffic_cone, barrier |
| 1 | 100×100 | ×2 | car, construction_vehicle |
| 2 | 50×50 | ×4 | truck, bus, trailer |

`max_levels` defaults to **3 (full FPN3)** and is used as-is in both the
standalone binary and the ROS2 `lidar_detection_node` — so all three levels
(including trucks/buses/trailers) are decoded in the live pipeline. The knob
exists to drop coarse levels (→ 2 or 1) and trade large-object recall for
memory if a deployment ever needs it, but nothing currently lowers it.

---

## Performance (Jetson Orin Nano Super)

```
                    KITTI data (~19K pts)    nuScenes data (~43K pts)
Voxelization      :  0.2ms                   0.2ms
Backbone+Head     : 12-15ms                  13-17ms  (TRT FP16, FPN3 3-level)
Decoder+NMS       :  6-11ms                  8-12ms   (per-level, merged)
Total (warm)      : 20-25ms  (~45 FPS)       25-30ms  (~37 FPS)
Total (1st frame) : ~140ms   (GPU warm-up)   ~140ms   (GPU warm-up)

Detections        : 6-58 per frame           140-191 per frame
                    (sparse suburban)        (dense urban, 10 classes)
```

---

## Setup

### One-command setup (recommended)

```bash
# Export FPN3 ONNX from Colab first (see fusion/train_pointpillars.py)
# Then copy to weights/:
cp pointpillars_nuscenes_fpn3.onnx ~/EdgeDrive-Perception/weights/

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
cp ../cuda-pointpillars-patches/tensorrt.cpp              src/common/tensorrt.cpp
cp ../cuda-pointpillars-patches/lidar-backbone.hpp        src/pointpillar/lidar-backbone.hpp
cp ../cuda-pointpillars-patches/lidar-backbone.cu         src/pointpillar/lidar-backbone.cu
cp ../cuda-pointpillars-patches/pointpillar.hpp           src/pointpillar/pointpillar.hpp
cp ../cuda-pointpillars-patches/pointpillar.cpp           src/pointpillar/pointpillar.cpp
cp ../cuda-pointpillars-patches/main.cpp                  src/main.cpp
cp ../cuda-pointpillars-patches/lidar-postprocess.hpp     src/pointpillar/lidar-postprocess.hpp
cp ../cuda-pointpillars-patches/lidar-postprocess.cu      src/pointpillar/lidar-postprocess.cu
```

### 3. Build

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
```

### 4. Build TRT FPN3 engine from ONNX

```bash
# Export FPN3 ONNX from mmdetection3d (see fusion/train_pointpillars.py)
# Copy to model/:
cp pointpillars_nuscenes_fpn3.onnx ~/EdgeDrive-Perception/cuda-pointpillars/model/

export TensorRT_Bin=/usr/src/tensorrt/bin/

$TensorRT_Bin/trtexec \
    --onnx=/home/gabriel/EdgeDrive-Perception/cuda-pointpillars/model/pointpillars_nuscenes_fpn3.onnx \
    --fp16 \
    --plugins=/home/gabriel/EdgeDrive-Perception/cuda-pointpillars/build/libpointpillar_core.so \
    --saveEngine=/home/gabriel/EdgeDrive-Perception/cuda-pointpillars/model/pointpillar_fpn3.plan \
    --verbose 2>&1 | tail -5

# Expected: &&&& PASSED
```

### 5. Run

```bash
# Copy nuScenes LiDAR frame to data/
cp /data/sets/nuscenes/samples/LIDAR_TOP/<scene>.pcd.bin ../data/

./pointpillar ../data/ ../data/ --timer
# Expected output:
# [Backbone] FPN3 mode (10 bindings)
# Voxelization: 0.2ms | Backbone: 13-17ms | Decoder+NMS: 8-12ms
# Detections after NMS: 140-191
```

---

## nuScenes Model Parameters

```
ONNX model      : pointpillars_nuscenes_fpn3.onnx (9 outputs, all 3 FPN levels)
Config          : pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py (mmdetection3d)
Checkpoint      : pointpillars_nuscenes.pth

Architecture    : HardVFE → PointPillarsScatter → SECOND → FPN3 → Anchor3DHead
Exported layers : SECOND + FPN + Anchor3DHead only (backbone+neck+head)
                  VFE and scatter run in C++ CUDA kernels

Point cloud range : [-50, -50, -5, 50, 50, 3] meters
Voxel size        : [0.25, 0.25, 8.0] meters
Grid size         : 400 × 400
Pillar features   : 64 channels
Point features    : 4 (x, y, z, intensity) — ring channel dropped
Max voxels        : 30,000
Max points/voxel  : 32

Classes (10, mmdetection3d ordering confirmed via inference_detector):
  0:car  1:truck  2:construction_vehicle  3:bus  4:trailer
  5:barrier  6:motorcycle  7:pedestrian  8:bicycle  9:traffic_cone

Anchors (8 per location):
  car-ish      [2.60×0.87×1.0] rot=[0°, 90°]
  pedestrian   [1.73×0.58×1.0] rot=[0°, 90°]
  bicycle      [1.00×1.00×1.0] rot=[0°, 90°]
  cone/barrier [0.40×0.40×1.0] rot=[0°, 90°]

Score threshold : 0.05 (standalone), 0.15 (ROS2)
NMS threshold   : 0.20
NMS pre-filter  : top-1000 by score before NMS (matches nuScenes test_cfg)
BBox code size  : 9 (DeltaXYZWLHRBBoxCoder)
```

---

## NMS Crash Fixes

Several NMS crashes were encountered and fixed during Jetson deployment:

**Zero-detection guard** — `nms_launch` crashes when `bndbox_num_=0` (0 CUDA grid blocks).
Added early return before sort+NMS if no detections survive threshold.

**h_mask_ OOM** — Original allocates `det_num_ × DIVUP(det_num_, 64) × 8` bytes.
For level 0 (det_num_=320,000) this was 12.8GB. Fixed by capping at nms_pre=1000:
`h_mask_size_ = 1000 × 16 × 8 = 128KB`.

**nms_pre cap** — After sort, cap `bndbox_num_` at 1000 before NMS.
Matches nuScenes `test_cfg nms_pre=1000` and prevents OOM on dense frames.

**Safety cap type** — `bndbox_num_` is unsigned; `<= 0` comparison always false.
Changed to `> det_num_` for the upper bounds check only.

---

## Known Issues

**`CUDASM` env var not persisted across reboots:**
`export CUDASM=87` must be set before `cmake`. Add to `~/.bashrc`:
```bash
echo "export CUDASM=87" >> ~/.bashrc
```

**High detection count (~140-191) on nuScenes:**
The model is running correctly. nuScenes scenes are dense urban environments.
Raise `score_thresh` to 0.3+ to reduce to most confident detections only.

**First frame slower (~50ms) than subsequent (~25ms):**
GPU warm-up effect — TRT kernel compilation on first inference.
Normal behaviour, not a bug.
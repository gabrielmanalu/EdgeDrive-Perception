# Fusion

Camera-LiDAR multi-modal perception pipeline for autonomous driving.
Covers LiDAR 3D object detection with PointPillars and Camera-LiDAR
late fusion in Bird's Eye View space.

---

## PointPillars LiDAR 3D Detection

### What PointPillars Does

LiDAR produces an unstructured 3D point cloud — thousands of points
with no fixed grid, which standard CNNs cannot process directly.
PointPillars solves this by converting the point cloud into a
**pseudo-image** using vertical columns (pillars) on the X-Y plane:

```
LiDAR point cloud (N × 4)         Pseudo-image (C × H × W)

   z↑    · ·                       ┌─────────────────────┐
    |  ·   · ·  ← car              │  feature grid       │
    | · ·      ·                   │  (like a 2D image)  │
    └────────── x    →  pillar  →  └─────────────────────┘
                        encoding        ↓ 2D CNN backbone
 irregular 3D points               3D bounding boxes output
```

This makes it fast, edge-friendly, and deployable on Jetson with TensorRT.

---

### BEV Visualizations

Three visualization modes are provided in `bev_visualization.py`:

#### 1. Basic BEV — Detection Boxes Only

Fast inspection of PointPillars output. Rotated 3D boxes projected
top-down with class labels and confidence scores.

![Basic BEV](../demo/screenshots/fusion/bev_detections.png)

---

#### 2. Side-by-Side — Raw Point Cloud vs Detections

Left: raw LiDAR scan colored by height (purple=ground, cyan=objects).
Right: semantic 3D boxes extracted by PointPillars.

> *"Left is what the LiDAR sees. Right is what PointPillars understands."*

![Point Cloud vs Detections](../demo/screenshots/fusion/pointcloud_vs_detections.png)

---

#### 3. Combined — Point Cloud Background with Detections Overlaid

Most visually impressive output. LiDAR points as background,
3D boxes with heading lines, range rings, and forward direction arrow.

![Combined BEV](../demo/screenshots/fusion/bev_with_pointcloud.png)

**Key elements:**
- **Colored points** — height-coded LiDAR scan (plasma colormap)
- **Range rings** — 10/20/30/40/50m distance markers
- **Rotated boxes** — correct heading angles per object
- **Heading lines** — small line from center to front edge of each box
- **Forward arrow** — ego vehicle driving direction (Y axis)
- **EGO rectangle** — ego vehicle at origin

---

### Detection Results (nuScenes Mini, Single Frame)

```
Sample: scene-0061, first keyframe
Score threshold: 0.3

Detected objects:
  car                       3
  bus                       1
  bicycle                   4
  traffic_cone              13
  trailer                   1

Total: 24 detections above 0.3 threshold
```

---

### Evaluation — Published Benchmark Numbers

nuScenes Mini does **not** include LiDAR sweep data (0 previous sweeps
per sample). PointPillars was trained with 10 fused sweeps, so local
evaluation on Mini produces near-zero mAP. Published numbers from the
MMDetection3D model zoo are used for reporting:

| Metric | Value | Dataset |
|---|---|---|
| mAP | 0.354 | Full nuScenes val (6019 samples, 10 sweeps) |
| NDS | 0.476 | Full nuScenes val |

**Per-class AP (full nuScenes val):**

| Class | AP |
|---|---|
| car | 0.576 |
| pedestrian | 0.748 |
| traffic_cone | 0.563 |
| barrier | 0.553 |
| motorcycle | 0.389 |
| bus | 0.355 |
| truck | 0.293 |
| bicycle | 0.225 |

Checkpoint: `hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d`
Source: [MMDetection3D Model Zoo](https://github.com/open-mmlab/mmdetection3d/tree/main/configs/pointpillars)

---

### Coordinate Transform

MMDetection3D outputs boxes in the **ego vehicle frame** (origin = ego).
nuScenes evaluation requires boxes in the **global frame** (origin = map).

```
Ego frame → Global frame:

pos_global = ego_rotation.rotate(pos_ego) + ego_translation
yaw_global = yaw_ego + ego_rotation.yaw_pitch_roll[0]
rot_quat   = Quaternion(axis=[0,0,1], angle=yaw_global)
```

A simplified yaw-only rotation (atan2 from quaternion components)
was tried first but produced incorrect global positions. Full quaternion
rotation via pyquaternion is required for correct results.

---

### ONNX/TensorRT Export

MMDetection3D v1.4 moved ONNX export to a separate repo (`mmdeploy`).
Even with mmdeploy, PointPillars ONNX export **excludes voxelization**
(Stage 1) and post-processing (Stage 3):

```
Stage 1: Voxelization (pillars)   ← Python/C++, NOT in ONNX
Stage 2: Neural network           ← exported to ONNX ✅
Stage 3: Post-processing (NMS)    ← Python/C++, NOT in ONNX
```

For Jetson deployment, **NVIDIA CUDA-PointPillars** provides a complete
C++ TensorRT pipeline including CUDA voxelization kernel:
https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars

---

### Environment Requirements

```
PyTorch  : 2.1.0+cu121  ← must match mmcv pre-built wheel
mmcv     : 2.1.0
mmdet    : 3.2.0
mmdet3d  : 1.4.0
mmengine : 0.10.7
```

**Important:** PyTorch 2.6+ (CUDA 12.4/12.8) has no pre-built mmcv
wheels. Downgrade to PyTorch 2.1.0 + CUDA 12.1 for this module.

---

### Setup & Usage

```bash
# 1. Install dependencies (see train_pointpillars.py docstring)
pip install torch==2.1.0 torchvision==0.16.0 \
    --index-url https://download.pytorch.org/whl/cu121
pip install -U openmim
mim install mmengine
mim install mmcv==2.1.0
mim install "mmdet>=3.0.0,<3.3.0"
mim install "mmdet3d>=1.1.0"
pip install nuscenes-devkit open3d pyquaternion

# 2. Setup, download weights, prepare dataset
python train_pointpillars.py \
    --nuscenes_root /data/sets/nuscenes \
    --weights_dir   ./pointpillars_weights \
    --mmdet3d_dir   ./mmdetection3d

# 3. Single-frame inference + BEV visualization
python pointpillars_inference.py \
    --mode single \
    --nuscenes_root /data/sets/nuscenes \
    --checkpoint    ./pointpillars_weights/pointpillars_nuscenes.pth

# 4. Full val set inference + evaluation
python pointpillars_inference.py \
    --mode eval \
    --nuscenes_root /data/sets/nuscenes \
    --checkpoint    ./pointpillars_weights/pointpillars_nuscenes.pth

# 5. Generate all 3 BEV visualizations
python bev_visualization.py \
    --lidar_path /data/sets/nuscenes/samples/LIDAR_TOP/xxx.pcd.bin \
    --checkpoint ./pointpillars_weights/pointpillars_nuscenes.pth \
    --output_dir ./bev_outputs
```

---

## Camera-LiDAR Late Fusion

Fuses YOLO26n 2D camera detections with PointPillars 3D LiDAR
detections in Bird's Eye View space:

```
┌─────────────────────────────────────────────────┐
│              Late Fusion Module                  │
├──────────────────┬──────────────────────────────┤
│  YOLO26n Camera  │  PointPillars LiDAR           │
│  2D boxes        │  3D boxes (already in BEV)    │
│  ↓ project to BEV│                              │
│  (camera calib)  │                              │
├──────────────────┴──────────────────────────────┤
│  Distance-based matching in BEV                 │
│  → Fused detections, combined confidence        │
├─────────────────────────────────────────────────┤
│  BEV Visualization                              │
│  Blue=LiDAR | Green=Camera | Red=Fused         │
└─────────────────────────────────────────────────┘
```

Files: `camera_to_bev.py`, `late_fusion.py`, `fusion_evaluation.py`

---

## File Reference

| File | Purpose |
|---|---|
| `train_pointpillars.py` | MMDet3D setup, weight download, dataset prep |
| `pointpillars_inference.py` | Inference + nuScenes evaluation pipeline |
| `bev_visualization.py` | Three BEV visualization modes |
| `camera_to_bev.py` | Project YOLO26n 2D detections into BEV space |
| `late_fusion.py` | Distance-based Camera-LiDAR fusion in BEV |
| `fusion_evaluation.py` | Fusion vs single-modal comparison metrics |
# Training

Fine-tuning YOLO26n and YOLOv8n on nuScenes front-camera data for
autonomous driving object detection and instance segmentation.

---

## Models

| Model | Task | mAP50 | mAP50-95 | Mask mAP50 | Params | Epochs |
|---|---|---|---|---|---|---|
| YOLO26n-det | Detection | 0.558 | 0.343 | — | 2.5M | 100 |
| YOLO26n-seg | Det + Seg | 0.594 | 0.360 | 0.484 | 2.9M | 100 |
| YOLOv8n-det | Detection (baseline) | 0.671 | 0.409 | — | 3.2M | 100 |

> Trained on nuScenes Mini (323 train / 81 val images, CAM_FRONT only).
> Full nuScenes (28k samples) expected to yield mAP50 0.65+ for YOLO26n.

---

## Classes (8)

| ID | Class | Train instances | Notes |
|---|---|---|---|
| 0 | car | 1943 | Most common (52.2%) |
| 1 | pedestrian | 880 | |
| 2 | bicycle | 86 | |
| 3 | motorcycle | 134 | |
| 4 | bus | 137 | |
| 5 | truck | 138 | |
| 6 | traffic_cone | 0 | Not present in mini — full dataset required |
| 7 | barrier | 401 | |

---

## Dataset Setup

### 1. Download nuScenes Mini

Register at https://www.nuscenes.org and download `v1.0-mini` (~4GB).

Expected directory structure:
```
/data/sets/nuscenes/
├── samples/          ← keyframe images (CAM_FRONT, CAM_BACK, LiDAR, etc.)
├── sweeps/           ← full-rate frames between keyframes
├── maps/             ← HD map data
└── v1.0-mini/        ← JSON annotation files
    ├── scene.json
    ├── sample.json
    ├── sample_data.json
    ├── ego_pose.json
    ├── calibrated_sensor.json
    └── ...
```

### 2. Convert Annotations

**Detection labels** (bounding boxes for YOLO26n-det and YOLOv8n-det):
```bash
python convert_nuscenes_det.py \
    --nuscenes_root /data/sets/nuscenes \
    --output_dir ./data/nuscenes_det \
    --version v1.0-mini
```

**Segmentation labels** (convex hull polygons for YOLO26n-seg):
```bash
python convert_nuscenes_seg.py \
    --nuscenes_root /data/sets/nuscenes \
    --output_dir ./data/nuscenes_seg \
    --version v1.0-mini
```

Expected output:
```
data/nuscenes_det/
├── images/
│   ├── train/   (323 symlinks → nuScenes samples/)
│   └── val/     (81 symlinks)
├── labels/
│   ├── train/   (323 .txt files — YOLO bbox format)
│   └── val/     (81 .txt files)
└── nuscenes.yaml

data/nuscenes_seg/
├── images/      (symlinks to same images)
├── labels/      (polygon labels — convex hull format)
└── nuscenes_seg.yaml
```

### 3. Train

```bash
# YOLO26n detection (primary model)
python train_yolo26n.py \
    --data ./data/nuscenes_det/nuscenes.yaml \
    --project ./runs \
    --epochs 100

# YOLO26n segmentation
python train_yolo26n_seg.py \
    --data ./data/nuscenes_seg/nuscenes_seg.yaml \
    --project ./runs \
    --epochs 100

# YOLOv8n detection (baseline comparison)
python train_yolov8n.py \
    --data ./data/nuscenes_det/nuscenes.yaml \
    --project ./runs \
    --epochs 100
```

Training time: ~70 min per model on Tesla T4 (Google Colab).

---

## Label Conversion — Key Design Decisions

### Coordinate Transform (Critical)

nuScenes stores all 3D annotations in **global world coordinates**.
Projecting directly to the image plane produces degenerate full-image boxes.
The correct pipeline requires a 3-step transform before projection:

```
Global frame
  → Ego vehicle frame   subtract ego translation, rotate by ego quaternion inverse
  → Camera frame        subtract camera translation, rotate by camera quaternion inverse
  → Image plane         project using camera intrinsic matrix K
```

Skipping any step was the root cause of a bug during development where
all bounding boxes projected to `w=1.0, h=1.0` (full image coverage).

### Split Strategy (Sample-level, not Scene-level)

Splitting by scene (e.g. scenes 0–7 train, scenes 8–9 val) caused the
val set to have 0 non-empty label files. Some scenes have no front-camera
annotations for our 8 target classes.

**Fix:** Random 80/20 split at the **sample level** with `seed=42` ensures
both splits draw annotations from all scene types.

### Segmentation Labels — Convex Hull vs Rectangle

A simple bbox-to-rectangle conversion (all 4 corners at the same positions)
produces identical rectangular masks for all objects regardless of viewing
angle. Instead, we project all **8 corners of the 3D bounding box** to the
image plane and compute their convex hull:

- Car seen head-on → tall narrow rectangle
- Car seen at 45° → trapezoid
- Pedestrian → narrow vertical polygon

This produces approximate but geometrically meaningful silhouettes.

---

## Why YOLO26n vs YOLOv8n — Understanding the Results

YOLOv8n achieves higher mAP50 (0.671 vs 0.558) on nuScenes Mini. This is
expected and does not mean YOLO26n is a worse model.

**Why YOLOv8n wins on small data:**
- More mature COCO pretraining (released 2023, extensively refined)
- Traditional NMS-based head converges faster on 323 training images
- YOLO26's NMS-free head needs more data to learn confident class predictions

**Where YOLO26n wins at deployment (Jetson Orin Nano):**
- Lower TensorRT inference latency (NMS-free = no post-processing step)
- Tighter latency variance (P99–P50 spread)
- Better INT8 quantization robustness (less mAP drop at INT8)
- Simpler C++ decoder (no NMS implementation needed)
- Lower power consumption under sustained inference

The mAP gap closes significantly with more data. Full nuScenes (28k samples)
is expected to yield comparable or better results for YOLO26n.

See [docs/yolo26_vs_yolov8.md](../docs/yolo26_vs_yolov8.md) for full analysis.

---

## File Reference

| File | Purpose |
|---|---|
| `convert_nuscenes_det.py` | nuScenes → YOLO bbox labels |
| `convert_nuscenes_seg.py` | nuScenes → YOLO polygon labels (convex hull) |
| `train_yolo26n.py` | YOLO26n detection fine-tuning |
| `train_yolo26n_seg.py` | YOLO26n segmentation fine-tuning |
| `train_yolov8n.py` | YOLOv8n detection baseline |
| `export_all_formats.py` | Export to TensorRT / ONNX / TFLite |
| `quantize.py` | PTQ/QAT INT8 quantization |
| `prune.py` | Structured channel pruning |
| `requirements.txt` | Python dependencies |

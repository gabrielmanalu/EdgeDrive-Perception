# System Architecture

EdgeDrive Perception implements a full autonomous driving perception pipeline
optimized for edge deployment on Jetson Orin Nano Super (67 TOPS, ~$250).

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                      Sensor Inputs                              │
├───────────────────────┬─────────────────────────────────────────┤
│  Camera (CAM_FRONT)   │  LiDAR (LIDAR_TOP)                      │
│  1600×900 @ 12Hz      │  32-beam, 360° sweep                    │
└───────────┬───────────┴──────────────┬──────────────────────────┘
            │                          │
            ▼                          ▼
┌───────────────────┐      ┌────────────────────────┐
│  Camera Pipeline  │      │   LiDAR Pipeline       │
│                   │      │                        │
│  Letterbox resize │      │  Point cloud voxelize  │
│  640×640 input    │      │  Pillar feature extract│
│                   │      │                        │
│  YOLO26n-det      │      │  PointPillars          │
│  TRT INT8         │      │  (pre-trained weights) │
│  3.5ms / 193 FPS  │      │  mAP=0.354, NDS=0.476  │
│                   │      │                        │
│  [1,12,8400]      │      │  3D boxes in BEV       │
│  raw head output  │      │  ego coordinates       │
│                   │      │                        │
│  cxcywh decode    │      │  LiDAR → ego transform │
│  class-agnostic   │      │                        │
│  NMS (IoU=0.3)    │      │                        │
│                   │      │                        │
│  2D boxes (cam)   │      │  3D boxes → BEV (x,y)  │
│  + class + score  │      │  + class + score       │
└─────────┬─────────┘      └──────────┬─────────────┘
          │                           │
          ▼                           ▼
┌─────────────────────────────────────────────────────┐
│                 Late Fusion (BEV)                   │
│                                                     │
│  Ground plane projection: camera 2D → BEV (x,y)     │
│  Class-aware distance matching (12m threshold)      │
│  +5m penalty for class mismatch                     │
│  Deduplication + front-hemisphere filter            │
│                                                     │
│  Fused score = 0.6 × LiDAR + 0.4 × Camera           │
└─────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────┐
│              Detection Output                        │
│  Fused detections | LiDAR-only | Camera-only         │
│  Class + score + BEV position                        │
└─────────────────────────────────────────────────────┘
```

---

## Camera Pipeline Detail

### Preprocessing (C++ TRT pipeline)

```
Input frame (any resolution, e.g. 1600×900 nuScenes)
    │
    ▼ cv::resize() — aspect-ratio preserving
Resized (e.g. 640×360)
    │
    ▼ cv::copyTo() onto 640×640 gray canvas (value=114)
Letterboxed 640×640 BGR
    │
    ▼ cv::cvtColor(BGR→RGB)
640×640 RGB
    │
    ▼ pixel loop: HWC→CHW + normalize [0,255]→[0,1]
    ▼ write directly to cudaHostAlloc pinned buffer (UMA zero-copy)
input_host_[3 × 640 × 640] float32
```

### Model: YOLO26n-det

```
Architecture : YOLO26n (NMS-free detection head)
Parameters   : 2.5M
Input        : [1, 3, 640, 640] FLOAT
Output       : [1, 12, 8400] FLOAT (end2end disabled on JetPack 6)
               channels 0-3 : cx, cy, w, h (640×640 space)
               channels 4-11: class probabilities (sigmoid applied)
Anchors      : 8400 = 80×80 + 40×40 + 20×20 (P3/P4/P5)
Training     : nuScenes Mini, 323 images, 100 epochs, Tesla T4
```

### Postprocessing (C++ decoder)

```
[1, 12, 8400] output
    │
    ▼ cxcywh → xyxy conversion
    ▼ scale to original image coordinates
    ▼ filter by score threshold (default: 0.3)
    ▼ class-agnostic NMS (IoU threshold: 0.3)
    │
    ▼
std::vector<Detection> {x1,y1,x2,y2, score, class_id, class_name}
```

---

## LiDAR Pipeline Detail

### PointPillars

```
Input  : LIDAR_TOP sweep (32-beam, ~35k points/frame)
         [x, y, z, intensity, time] per point
Pillar size   : 0.2m × 0.2m
Max pillars   : 12000
Output : 3D boxes [x,y,z,w,l,h,yaw] in ego frame + class + score
```

### Coordinate Transforms

```
LiDAR frame → (calibration extrinsics) → Ego frame → BEV fusion
Ego frame   → (camera intrinsics K)    → Image coordinates
```

---

## Fusion Detail

See [`docs/sensor_fusion_analysis.md`](sensor_fusion_analysis.md) for full analysis.

```
BEV matching:
  Camera detections projected to ground plane (z=0 assumption)
  LiDAR detections in ego BEV (x,y)
  Distance matrix + class penalty (+5m mismatch)
  Greedy matching: threshold 12m
  Fused score = 0.6 × LiDAR + 0.4 × Camera
```

---

## Deployment Architecture

### Python (development/baseline)

```
Ultralytics YOLO wrapper → TRT engine → Python postprocessor
Bottleneck: Python GIL, preprocessing overhead (5.6ms/frame)
Throughput: 87.6 FPS (FP16), 78.5 FPS (INT8)
```

### C++ (production)

```
TRTEngine → cudaHostAlloc pinned UMA → enqueueV3() → YOLO26Decoder
Bottleneck: GPU-limited
Throughput: 193.3 FPS (INT8, 404 images)
```

### Jetson UMA Zero-Copy

```
Traditional discrete GPU:
  CPU RAM ──cudaMemcpy──▶ GPU VRAM   (~1-2ms overhead/frame)

Jetson UMA (this project):
  cudaHostAlloc(cudaHostAllocMapped)
  CPU writes input_host_ → GPU reads input_device_ (same physical memory)
  GPU writes output_device_ → CPU reads output_host_ (same physical memory)
  Zero cudaMemcpy() calls
```

---

## Quantization Pipeline

```
PyTorch FP32 (training baseline)
    │
    ├── Colab accuracy proof (TFLite):
    │     FP16 PTQ  → +0.36% mAP
    │     INT8 PTQ  → +0.45% mAP ← best
    │     INT8 QAT  → +0.32% mAP
    │
    └── Jetson performance (TensorRT):
          FP16 → 4.2ms TRT, 87.6 FPS (Python)
          INT8 → 3.5ms TRT, 193.3 FPS (C++) ← deployed
```

---

## Hardware

| Component | Spec |
|---|---|
| SoC | NVIDIA Orin (12-core ARM Cortex-A78AE + Ampere GPU) |
| GPU | 1024 CUDA cores, SM 8.7, 1017 MHz |
| Memory | 8GB LPDDR5 unified (CPU+GPU) |
| Storage | 512GB NVMe SSD |
| JetPack | R36.4.7 |
| TensorRT | 10.3.0 |
| Cost | ~$250 |
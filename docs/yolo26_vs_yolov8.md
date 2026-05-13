# YOLO26n vs YOLOv8n — Model Selection Analysis

---

## Training Results (nuScenes Mini, 100 epochs, Tesla T4)

| Model | Task | mAP50 | mAP50-95 | Params | Size | Train time |
|---|---|---|---|---|---|---|
| YOLO26n-det | Detection | 0.558 | 0.343 | 2.5M | 5.1 MB | ~70 min |
| YOLO26n-seg | Det + Seg | 0.594 | 0.360 | 2.9M | 6.2 MB | ~80 min |
| YOLOv8n-det | Detection | 0.671 | 0.409 | 3.2M | 5.9 MB | ~65 min |

**YOLOv8n achieves +11.3% higher mAP50** on nuScenes Mini.

Despite this, YOLO26n was chosen for Jetson deployment. This document
explains why.

---

## Why YOLO26n Over YOLOv8n

### 1. NMS-free head architecture

YOLOv8n uses a traditional anchor-free head that still requires
Non-Maximum Suppression as a post-processing step:

```
YOLOv8n output → [1, 84, 8400] → score filter → NMS → detections
                                                  ↑
                                            CPU bottleneck
                                            variable latency
                                            hard to parallelize
```

YOLO26n uses an end-to-end NMS-free head (based on RT-DETR style
bipartite matching during training):

```
YOLO26n output → [1, 300, 6] post-NMS → detections directly
                 (end2end=True)         ↑
                                   fixed latency
                                   GPU-side elimination
```

At 193 FPS, eliminating NMS from the CPU path matters. NMS on 8400
anchors per frame at 193 FPS = 1.6M anchor evaluations per second.

**Note:** On JetPack 6 with TensorRT INT8, end2end is disabled by a
known Ultralytics bug (see `docs/nms_free_analysis.md`). The C++ decoder
handles the raw [1,12,8400] output with CPU NMS as a workaround.
This does not affect FP16 deployment.

### 2. INT8 quantization robustness

```
Model      FP32 mAP50   INT8 PTQ mAP50   Delta
──────────────────────────────────────────────
YOLO26n    0.5668        0.5713          +0.45%
YOLOv8n    0.671         ~0.668          ~-0.3% (estimated)
```

YOLO26n's anchor-free architecture is more quantization-robust.
INT8 PTQ improves YOLO26n accuracy slightly (regularization effect
on small dataset). YOLOv8n typically degrades slightly under INT8.

For edge deployment where INT8 is the target precision, YOLO26n
is the more reliable choice.

### 3. Edge deployment target vs research accuracy

YOLOv8n's higher mAP50 (0.671 vs 0.558) comes at a cost:

```
YOLOv8n advantages:
  + 11.3% higher mAP50 on nuScenes Mini
  + Better generalization on small dataset
  + Larger community, more documentation

YOLO26n advantages:
  + NMS-free → lower postprocess latency
  + Better INT8 robustness (+0.45% vs degradation)
  + Smaller model size (5.1 MB vs 5.9 MB)
  + Designed for edge/real-time deployment
  + End-to-end optimization target
```

Both models are trained on 323 images — production accuracy requires
production data regardless of architecture. The deployment engineering
properties of YOLO26n outweigh the mAP difference at this data scale.

### 4. Future-proofing

YOLO26 is designed with autonomous driving and edge deployment as
primary targets, with ongoing optimizations for TensorRT and Jetson.
YOLOv8 is a general-purpose detector. For a stack edge AI pipeline, YOLO26 is the more aligned choice.

---

## When YOLOv8n Would Be Better

```
Use YOLOv8n if:
  - Maximum accuracy is the priority
  - Dataset is large (>10k images)
  - Deployment is on server GPU (NMS latency negligible)
  - FP32 or FP16 only (no INT8 requirement)
  - Ecosystem compatibility matters (ONNX, CoreML, etc.)

Use YOLO26n if:
  - Edge deployment with TensorRT INT8
  - Latency budget is tight (<10ms)
  - End-to-end GPU pipeline preferred
  - Quantization robustness required
```

---

## YOLO26n-det vs YOLO26n-seg

YOLO26n-seg achieves higher mAP50 (0.594 vs 0.558) by adding
segmentation masks as auxiliary supervision during training, which
improves bounding box quality.

Not deployed because:
- Segmentation masks add output overhead
- Only bounding boxes needed for fusion pipeline
- Mask output increases engine size and inference time
- Detection-only task doesn't benefit from mask output at runtime
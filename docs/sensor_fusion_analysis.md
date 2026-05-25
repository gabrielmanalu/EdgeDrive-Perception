# Sensor Fusion Analysis — EdgeDrive Perception

Camera-LiDAR late fusion design, implementation, and results on nuScenes.

---

## Approach: Late Fusion

Two fusion strategies were considered:

```
Early/BEV fusion (e.g. BEVFusion):
  Pros: shared feature space, state-of-the-art mAP
  Cons: ~200MB model, ~5 FPS on Jetson Orin Nano Super (67 TOPS)
  Verdict: not viable for real-time edge deployment

Late fusion (this project):
  Pros: 180 FPS camera + 37-45 FPS LiDAR → real-time on edge hardware
        each modality independently debuggable
        modular: swap either detector without retraining
  Cons: no shared feature learning, BEV projection approximation
  Verdict: correct tradeoff for this hardware target
```

---

## Pipeline

```
Camera (YOLO26n-det TRT INT8):    LiDAR (PointPillars FPN3 TRT FP16):
  2D boxes (u, v) in image px       3D boxes (x,y,z,w,l,h,yaw)
  ~180 FPS, nuScenes CAM_FRONT      in LiDAR sensor frame
        │                                │
        ▼                                ▼
  Ground plane projection          LiDAR→Ego transform
  (u,v) → (Z,X) via K^-1          ego_forward = -lidar_Y
  Z = h × fy / (v - cy)           ego_left    = +lidar_X
  X = (u - cx) × Z / fx           (from calibrated_sensor quaternion)
        │                                │
        └──────────────┬─────────────────┘
                       ▼
              BEV Distance Matching
              D[i,j] = euclidean(cam_bev[i], lidar_bev[j])
              threshold: 8m
              assignment: Hungarian (greedy)
                       │
                       ▼
              Fused score = 0.6 × lidar + 0.4 × camera
              Class label = camera class (higher semantic confidence)
                       │
              ┌────────┼────────┐
              ▼        ▼        ▼
            RED      GREEN    BLUE
           Fused   Cam only  LiDAR only
```

---

## Camera BEV Projection

**Intrinsics (nuScenes CAM_FRONT):**
```
fx = fy = 1266.417
cx = 816.267
cy = 491.507
camera height = 1.5m (approximate ground plane)
```

**Ground plane assumption:**
```
v_bottom = bottom edge of 2D bounding box (ground contact point)
Z = h × fy / (v_bottom - cy)   ← forward distance
X = (u - cx) × Z / fx           ← lateral offset
BEV position = (Z, -X)          ← (forward, left) in ego frame
Valid range: Z ∈ [1m, 60m]
```

**Limitation:** Accuracy degrades for objects not on the ground plane (trucks,
buses, elevated objects). Lateral error grows with distance (~15m error at 35m
range). Angular matching compensates for this — see matching section.

---

## LiDAR Coordinate Transform

**nuScenes LIDAR_TOP calibrated_sensor:**
```
translation: [0.943713, 0.0, 1.84023]  ← 1.84m above ground
quaternion:  [0.7077955, -0.006492, 0.010646, -0.7063073]

Rotation matrix:
  LiDAR +X → ego: (0.002, -1.0,  0.0)  ← ego RIGHT
  LiDAR +Y → ego: (1.0,   0.002, 0.0)  ← ego FORWARD

Transform applied in lidar_detection_node:
  ego_forward = b.y
  ego_left    = -b.x
```

This was verified empirically: raw LiDAR output `b.y=25.3` for a car 25m ahead
confirms LiDAR +Y = ego forward.

`Detection3DArray`is the canonical LiDAR output used by fusion and is transformed into base_link as x = b.y, y = -b.x. The standalone `LiDAR MarkerArray` publisher is a debug visualization of raw PointPillars output and is not consumed by fusion.

---

## Matching Results — ROS2 Live (nuScenes scene0, 39 frames)

```
camera_node     : YOLO26n TRT INT8 → 2–11 detections/frame (barriers excluded)
lidar_node      : CUDA-PP FPN3 FP16 → 0–16 detections/frame (score ≥ 0.15)
fusion_node     : BEV distance matching, 8m threshold

Typical frame breakdown:
  cam=6 lidar=4 matched=3 fused=4   ← 50% cam matched
  cam=6 lidar=8 matched=6 fused=8   ← 100% cam matched
  cam=2 lidar=5 matched=2 fused=5   ← 100% cam matched
  cam=5 lidar=3 matched=0 fused=3   ← 0% (LiDAR detections behind ego)

Match rate: 40–100% of camera detections per frame (when LiDAR has detections)
```

---

## Debugging Trail

Several coordinate system bugs were found and fixed during development:

**1. Single FPN level (original)**
Only level 0 (200×200, small anchors) was exported from mmdetection3d.
Cars were not detected — they appear at level 1 (100×100, ×2 anchors).
Fix: export all 3 FPN levels; confirmed by comparing standalone output
(67-104 cars/frame) vs original (0 cars, only pedestrians mislabeled as bicycle).

**2. Class ordering mismatch**
Single-level model labeled all detections as class 7 = "bicycle".
After FPN3 export and Colab `inference_detector` verification:
  - class 0 = car ✅, class 7 = pedestrian ✅, class 8 = bicycle ✅
Fix: updated `NUSCENES_CLASSES[]` array ordering in lidar_detection_node.

**3. LiDAR coordinate frame**
Multiple wrong sign combinations tried. Final correct transform derived from:
  - Calibrated sensor quaternion → rotation matrix
  - Empirical verification: `b.y=25.3` → 25m ahead confirmed ✅
Fix: `ego_forward = b.y, ego_left = -b.x`

**4. Double coordinate transform**
fusion_node was applying a second transform on top of lidar_detection_node's
already-transformed coordinates, giving 40–60m distances for nearby objects.
Fix: fusion_node reads Detection3DArray position directly (already in ego frame).

**5. Barrier noise**
Camera detecting 5–15 barriers/frame, cluttering BEV and reducing match quality.
Fix: barriers filtered from annotated image, Detection2DArray, and camera_markers.

---

## Colab vs ROS2 Fusion

Two implementations of late fusion exist in this project:

| | Colab | ROS2 |
|---|---|---|
| Location | `fusion/` notebooks | `ros2_ws/src/.../fusion_node.cpp` |
| Data | nuScenes sample data | nuScenes bag replay |
| Camera | mmdetection2d YOLO | YOLO26n TRT INT8 |
| LiDAR | mmdetection3d PointPillars | CUDA-PointPillars FPN3 |
| Matching | BEV distance + class penalty | BEV distance 8m |
| Threshold | 12m | 8m |
| Visualization | matplotlib | RViz2 MarkerArray |
| Real-time | No | Yes (~2Hz bag rate) |

---

## Parameter Choices

**match_threshold = 8m:**
Camera BEV projection error at 16m forward ≈ 4–8m lateral (ground plane approx).
Threshold set to cover typical projection error without generating false positives.
Class-agnostic matching used — camera class label applied to fused result.

**score weights = 0.6 LiDAR + 0.4 Camera:**
LiDAR has accurate 3D position (metric space), camera has higher semantic confidence.
Weights reflect LiDAR's superior localization while benefiting from camera's class labels.

**sync_tolerance = 2.0s:**
Bag produces perfectly synchronized timestamps (same nuScenes sample).
Large tolerance is safe and avoids dropped frames from minor timing jitter.

**score_threshold = 0.15 (LiDAR), 0.30 (Camera):**
LiDAR threshold lower to increase recall for fusion matching.
Camera threshold higher to reduce false-positive barriers/duplicates.
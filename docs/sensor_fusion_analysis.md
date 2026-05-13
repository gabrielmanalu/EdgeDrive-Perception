# Sensor Fusion Analysis

Camera-LiDAR late fusion results on nuScenes Mini sample data.

---

## Approach: Late Fusion

Two fusion strategies were considered:

```
Early/BEV fusion (e.g. BEVFusion):
  Pros: shared feature space, state-of-the-art mAP
  Cons: ~200MB model, ~5 FPS on Jetson Orin Nano Super
  Verdict: not viable for real-time edge deployment

Late fusion (this project):
  Pros: 193 FPS camera + fast LiDAR → near real-time
        each modality independently debuggable
        modular: swap either detector without retraining
  Cons: no shared feature learning, BEV projection approximation
  Verdict: correct tradeoff for this hardware target
```

---

## Pipeline

```
Camera (YOLO26n-det):           LiDAR (PointPillars):
  2D boxes (x1,y1,x2,y2)         3D boxes (x,y,z,w,l,h,yaw)
  in image coordinates            in ego/vehicle coordinates
        │                                │
        ▼                                ▼
  Ground plane projection          BEV projection (x,y)
  (u,v) → (x,y) via K^-1          using box center
  z = 0 assumption                 
        │                                │
        └──────────────┬─────────────────┘
                       ▼
              BEV Distance Matching
              D[i,j] = euclidean(cam[i], lidar[j])
              + 5m penalty if class mismatch
              threshold: 12m
                       │
                       ▼
              Fused score = 0.6 × lidar + 0.4 × camera
              (LiDAR weighted higher: metric accuracy)
                       │
              ┌────────┼────────┐
              ▼        ▼        ▼
           Fused    LiDAR   Camera
           dets     only    only
```

---

## Sample Results (nuScenes Mini, sample[1])

```
Camera detections (YOLO26n-det):  9 objects
LiDAR detections (PointPillars): 13 objects

Fusion output:
  Fused (matched)    : 13 detections
  LiDAR-only         :  5 detections  ← occluded from camera
  Camera-only        :  4 detections  ← below LiDAR range or low confidence
  Total output       : 22 detections
```

---

## Fusion Parameter Choices

### Distance threshold: 12m

```
Tested: 5m, 8m, 12m, 20m on sample[1]

5m:  too strict → misses valid matches (camera BEV projection error)
8m:  better, still misses some at range
12m: best match count, no obvious false matches
20m: too permissive → false matches at range
```

BEV projection via K^-1 accumulates error at distance. A 30m object
with 5% projection error = 1.5m error → 12m threshold provides margin.

### Class mismatch penalty: +5m

```
Rationale: same-class objects should match preferentially,
but cross-class matches are not forbidden (e.g. PointPillars
predicts "truck", camera predicts "bus" for same object).

+5m penalty: allows cross-class match if objects are very close
             in BEV, but prefers same-class matches.
```

### Score weighting: 0.6 LiDAR / 0.4 Camera

```
LiDAR advantage: metric distance accuracy, not affected by lighting
Camera advantage: class discrimination, texture features

LiDAR weighted higher because:
  - BEV position accuracy is primary for downstream planning
  - Camera BEV projection has z=0 approximation error
  - PointPillars trained on full nuScenes val (more data)
```

---

## Limitations

**Ground plane assumption (z=0):**
Camera 2D boxes are projected to BEV assuming z=0 (flat road).
This introduces error for:
- Elevated objects (trucks, buses above road level)
- Objects on slopes or ramps
- Camera height variations

**Single sweep LiDAR:**
nuScenes Mini provides single-sweep point clouds (~35k points).
Full nuScenes uses 10-sweep accumulation for denser coverage.
Single sweep has gaps for distant or fast-moving objects.

**Offline validation only:**
Fusion validated on nuScenes sample data in Colab. Live fusion
requires calibrated camera-LiDAR extrinsics for the specific
sensor mounting configuration. The C++ port (`late_fusion.cpp`)
and ROS2 bag replay are planned for integration phase.

**Class vocabulary mismatch:**
Camera model: 8 nuScenes classes (trained on Mini)
PointPillars: 10 nuScenes classes (full val set)
Classes not in camera vocabulary (trailer, construction_vehicle)
appear as LiDAR-only detections.

---

## Files

```
fusion/
  camera_to_bev.py       ← ground plane projection
  late_fusion.py         ← matching + scoring logic
  fusion_evaluation.py   ← evaluation on nuScenes samples
  README.md              ← fusion module documentation

deployment/src/
  late_fusion.cpp        ← C++ port (in progress)
```
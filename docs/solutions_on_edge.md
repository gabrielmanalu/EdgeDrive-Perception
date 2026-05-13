# Ultralytics Solutions on Edge

Ultralytics Solutions API provides ready-made computer vision applications
built on top of YOLO detection. This document evaluates their applicability
to edge deployment and autonomous driving.

---

## Solutions Evaluated

Five solutions were run on nuScenes footage using YOLO26n-det as the
underlying detector:

```
1. ObjectCounter   — count objects crossing a defined line
2. Heatmap         — pixel-level detection frequency map
3. SpeedEstimator  — track objects, estimate velocity
4. RegionCounter   — count objects within a defined polygon
5. DistanceCalc    — estimate distance between detected objects
```

---

## Speed Estimation

### Method

Ultralytics SpeedEstimator uses object tracking (ByteTrack) to measure
pixel displacement between frames, then converts to real-world speed
via a pixel-per-meter calibration factor.

```
For each tracked object:
  pixel_displacement = track[t].center - track[t-1].center (pixels)
  dt = 1 / fps  (time between frames)
  speed_px_per_s = pixel_displacement / dt

  speed_kmh = speed_px_per_s × (real_world_distance / pixel_distance) × 3.6
              ↑ calibration factor — camera-specific
```

### Limitations on Dashcam Footage

**Perspective distortion:**
Objects near the horizon occupy fewer pixels per meter than objects
close to the camera. A fixed pixel-per-meter calibration is only
accurate at one distance.

```
Object at 5m:  ~50 px/m  (large, fills frame)
Object at 30m: ~8 px/m   (small, near horizon)
```

**Stationary camera assumption:**
Speed estimation assumes a stationary camera. For ego-vehicle motion,
all detected objects appear to move relative to the camera — even
stationary objects (lane markings, barriers, parked cars).

Without subtracting ego-vehicle velocity, the estimator measures
*relative velocity* (object speed minus ego speed), not absolute speed.

```
Ego vehicle: 40 km/h
Parked car (absolute speed: 0 km/h)
Estimated speed: ~40 km/h (relative) ← wrong
```

**Practical use case:**
Speed estimation from a moving vehicle requires ego odometry or IMU
integration to subtract ego motion. This is planned as part of the
ROS2 integration phase where `/odom` topic provides ego velocity.

For now, SpeedEstimator is validated on stationary camera footage
only (e.g. intersection monitoring), not dashcam.

---

## Object Counting

ObjectCounter and RegionCounter work reliably on nuScenes footage for
stationary zones. Useful for:

```
Intersection analysis  : count vehicles/pedestrians per cycle
Parking lot monitoring : occupancy counting
Zone intrusion         : alert when object enters defined region
```

Not directly applicable to moving vehicle use case without
coordinate transformation to world frame.

---

## Heatmap

Heatmap shows detection frequency per pixel over a sequence.
On nuScenes CAM_FRONT:

```
High frequency zones:
  Road surface (center-bottom)  ← cars, always present
  Sidewalk regions              ← pedestrians
  Horizon line                  ← distant vehicles

Low frequency zones:
  Sky, buildings                ← no detections
```

Useful for: identifying blind spots, evaluating dataset coverage,
verifying detector consistency over time.

---

## Distance Calculation

DistanceCalc estimates pixel-distance between detected bounding box
centers. Converts to real-world distance using camera intrinsics and
assumed object height.

```
Limitations:
  - Only accurate for objects on the same ground plane
  - Error increases with perspective distortion
  - No depth sensing (monocular camera)

For actual distance estimation, LiDAR range measurements from
PointPillars provide metric accuracy without the above limitations.
DistanceCalc is a camera-only fallback.
```

---

## Edge Performance

All solutions run on top of standard YOLO26n-det inference.
Additional overhead per solution:

```
ObjectCounter  : +0.2ms  (line crossing check)
Heatmap        : +1.5ms  (pixel accumulation, frame resize)
SpeedEstimator : +0.8ms  (ByteTrack update, speed calc)
RegionCounter  : +0.3ms  (polygon point-in-polygon check)
DistanceCalc   : +0.1ms  (center distance, trivial)
```

All solutions remain real-time at 30+ FPS even with overhead.
ByteTrack (SpeedEstimator) is the most expensive due to Kalman filter
state updates per tracked object.

---

## Relevance to Autonomous Driving

```
Solution         AD Use Case                    Status
─────────────────────────────────────────────────────
ObjectCounter    traffic density estimation     useful
Heatmap          sensor coverage analysis       useful
SpeedEstimator   relative velocity (w/ ego)     partial ⚠️
RegionCounter    zone monitoring                useful
DistanceCalc     camera fallback distance       limited
```

Production autonomous driving uses dedicated modules for each:
velocity from Kalman-filtered tracks + IMU, range from LiDAR,
occupancy from HD maps. Solutions API is a rapid prototyping tool,
not a production component.
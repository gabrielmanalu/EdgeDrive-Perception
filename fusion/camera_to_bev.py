"""
Camera-to-BEV Projection
========================
Projects YOLO26n 2D bounding box detections from the front camera
image into Bird's Eye View (BEV) ego coordinates using the ground
plane intersection method.

How it works:
    1. Take the bottom-center pixel of each 2D bounding box
       (this is the ground contact point of the object)
    2. Unproject the pixel to a 3D ray in camera frame using K_inv
    3. Intersect the ray with the ground plane z=0 in camera frame
       Scale factor: t = camera_height / ray[1]  (y-axis points down)
    4. Transform the 3D point from camera frame to ego vehicle frame
       using the calibrated sensor extrinsics (rotation + translation)
    5. Return (x, y) in ego BEV frame (meters)

Coordinate frames:
    Image frame  : (u, v) in pixels, origin top-left
    Camera frame : x=right, y=down, z=forward (OpenCV convention)
    Ego frame    : x=forward, y=left, z=up (nuScenes convention)

nuScenes CAM_FRONT calibration (used for development):
    fx = fy = 1266.4 pixels
    cx = 816.27, cy = 491.51
    Camera height above ground: 1.51m
    Camera->Ego translation: [1.70, 0.016, 1.51] meters

Known limitations:
    1. Ground plane assumption (z=0):
       Fails for objects partially occluded at the base (e.g. cars
       behind barriers), objects on elevated roads, or very distant
       objects (>50m). These produce large y-axis errors (up to 25m).

    2. Monocular depth ambiguity:
       Camera provides no direct depth measurement. BEV position is
       estimated from the geometric constraint that objects touch the
       ground. This is approximate, not measured.

    3. Single camera (CAM_FRONT only):
       nuScenes has 6 cameras (360 degree coverage). This implementation
       uses only the front camera. Full coverage requires running the
       same pipeline for all 6 cameras and merging.

    In real fusion, LiDAR depth overrides camera BEV estimates for
    matched objects. Camera BEV is used only for objects LiDAR misses
    (thin objects like barriers partially visible only to camera).

C++ deployment note:
    This Python implementation is the reference for the C++ port in
    deployment/src/camera_to_bev.cpp. The math is identical:
    matrix inversion of K, quaternion rotation, vector addition.

Usage:
    from camera_to_bev import run_camera_to_bev, bbox_to_bev

    detections, img_path = run_camera_to_bev(
        nusc, model, sample_idx=1, score_thresh=0.3
    )
    for d in detections:
        print(f"{d['class_name']} at {d['bev_xy']} ({d['distance']:.1f}m)")
"""

import os
import numpy as np
from pyquaternion import Quaternion
from ultralytics import YOLO
from nuscenes.nuscenes import NuScenes


# ── nuScenes class names (YOLO26n output order) ───────────────────────────────

YOLO_CLASS_NAMES = [
    'car', 'pedestrian', 'bicycle', 'motorcycle',
    'bus', 'truck', 'traffic_cone', 'barrier'
]

# Maximum BEV distance to keep a detection (meters)
# Beyond this, ground plane errors become too large to be useful
MAX_PROJECTION_DIST = 60.0


# ── Core projection functions ─────────────────────────────────────────────────

def bbox_to_bev(bbox_xyxy, K, cam_translation, cam_rotation,
                camera_height=1.51):
    """
    Project a 2D bounding box bottom-center pixel to BEV ego coordinates
    using the ground plane intersection method.

    The bottom-center pixel (u, y_bottom) represents the point where
    the object contacts the ground — the most geometrically stable
    pixel for depth estimation via ground plane intersection.

    Args:
        bbox_xyxy      : np.ndarray [x1, y1, x2, y2] in pixels
        K              : np.ndarray (3x3) camera intrinsic matrix
        cam_translation: list [x, y, z] camera position in ego frame (m)
        cam_rotation   : list [w, x, y, z] quaternion camera->ego
        camera_height  : float, camera height above ground plane (m)
                         nuScenes CAM_FRONT z = 1.51m

    Returns:
        np.ndarray [x, y] in ego BEV frame (meters), or None if invalid
        x = forward (positive = ahead of ego)
        y = left    (positive = left of ego, negative = right)
    """
    x1, y1, x2, y2 = bbox_xyxy

    # Bottom-center pixel — ground contact point of the object
    u = (x1 + x2) / 2.0
    v = y2

    # Step 1: Unproject pixel to normalized ray in camera frame
    K_inv   = np.linalg.inv(K)
    ray_cam = K_inv @ np.array([u, v, 1.0])

    # Step 2: Intersect ray with ground plane
    # Camera frame: y axis points DOWN
    # Camera is at height h above ground
    # Ground plane in camera frame: y = camera_height
    # Solve: ray_cam[1] * t = camera_height
    if ray_cam[1] <= 0:
        # Ray points upward — no ground intersection
        return None

    t = camera_height / ray_cam[1]
    if t <= 0:
        return None

    # Step 3: 3D point in camera frame
    point_cam = ray_cam * t

    # Step 4: Transform camera frame -> ego frame
    R_cam2ego = Quaternion(cam_rotation).rotation_matrix
    t_cam2ego = np.array(cam_translation)
    point_ego = R_cam2ego @ point_cam + t_cam2ego

    return point_ego[:2]  # [x_ego, y_ego]


def run_camera_to_bev(nusc, model, sample_idx=0,
                      score_thresh=0.3, camera_height=1.51,
                      max_dist=MAX_PROJECTION_DIST):
    """
    Run YOLO26n inference on a nuScenes front camera image and
    project all detections to BEV ego coordinates.

    Args:
        nusc         : NuScenes instance
        model        : loaded Ultralytics YOLO model
        sample_idx   : index into nusc.sample list
        score_thresh : minimum confidence threshold
        camera_height: camera height above ground (meters)
        max_dist     : maximum BEV projection distance to keep (meters)

    Returns:
        detections : list of dicts with keys:
            class_name, score, bbox_px, bev_xy, distance
        img_path   : str, path to the camera image
    """
    sample    = nusc.sample[sample_idx]
    cam_token = sample['data']['CAM_FRONT']
    cam_data  = nusc.get('sample_data', cam_token)
    cs        = nusc.get('calibrated_sensor',
                         cam_data['calibrated_sensor_token'])

    img_path = os.path.join(nusc.dataroot, cam_data['filename'])

    K               = np.array(cs['camera_intrinsic'])
    cam_translation = cs['translation']
    cam_rotation    = cs['rotation']

    results = model(img_path, verbose=False)[0]

    detections = []
    for box in results.boxes:
        score = float(box.conf[0])
        if score < score_thresh:
            continue

        label    = int(box.cls[0])
        cls_name = (YOLO_CLASS_NAMES[label]
                    if label < len(YOLO_CLASS_NAMES) else 'unknown')
        bbox     = box.xyxy[0].cpu().numpy()

        bev_xy = bbox_to_bev(
            bbox, K, cam_translation, cam_rotation, camera_height
        )
        if bev_xy is None:
            continue

        distance = float(np.sqrt(bev_xy[0]**2 + bev_xy[1]**2))
        if distance > max_dist:
            continue

        detections.append({
            'class_name': cls_name,
            'score':      score,
            'bbox_px':    bbox,
            'bev_xy':     bev_xy,
            'distance':   distance,
        })

    return detections, img_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Project YOLO26n camera detections to BEV'
    )
    parser.add_argument('--nuscenes_root', default='/data/sets/nuscenes')
    parser.add_argument('--weights',
                        default='/content/drive/MyDrive/yolo_runs/'
                                'yolo26n_nuscenes/weights/best.pt')
    parser.add_argument('--sample_idx',   type=int,   default=1)
    parser.add_argument('--score_thresh', type=float, default=0.3)
    args = parser.parse_args()

    nusc  = NuScenes(version='v1.0-mini',
                     dataroot=args.nuscenes_root, verbose=False)
    model = YOLO(args.weights)

    detections, img_path = run_camera_to_bev(
        nusc, model,
        sample_idx=args.sample_idx,
        score_thresh=args.score_thresh
    )

    print(f"Sample {args.sample_idx}: {len(detections)} detections projected to BEV")
    print(f"\n{'Class':<15} {'Score':>6} {'x':>8} {'y':>8} {'Dist':>8}")
    print("-" * 50)
    for d in sorted(detections, key=lambda x: x['distance']):
        print(f"{d['class_name']:<15} {d['score']:>6.3f} "
              f"{d['bev_xy'][0]:>8.2f} {d['bev_xy'][1]:>8.2f} "
              f"{d['distance']:>7.1f}m")
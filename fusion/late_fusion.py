"""
Camera-LiDAR Late Fusion in BEV
================================
Fuses YOLO26n camera detections with PointPillars LiDAR detections
in Bird's Eye View space using distance-based greedy matching.

What is late fusion:
    Each modality (camera, LiDAR) runs its own detection pipeline
    independently. Late fusion combines their outputs by matching
    detections that refer to the same physical object.

    Camera -> YOLO26n -> 2D boxes -> BEV projection -> (x, y)
    LiDAR  -> PointPillars -> 3D boxes -> ego frame -> (x, y)
    Both   -> distance matching -> fused detections

Why late fusion vs early/deep fusion:
    Late fusion is modular — each sensor can fail independently
    without breaking the other. It's also more interpretable
    (you can see exactly what each modality detected) and
    computationally lighter for edge deployment.

    Early/deep fusion (e.g. BEVFusion) feeds both modalities
    into a single neural network for better accuracy, but
    requires retraining on a large multi-modal dataset.

Matching algorithm (v3 — class-aware optimal matching):
    1. Build all valid candidate pairs (camera, LiDAR) within
       match_thresh distance and compatible classes
    2. Add class penalty (+5m) for cross-class matches
       (e.g. truck<->car) to prefer same-class matches
    3. Sort candidates by penalized distance score
    4. Greedy match: assign closest pairs first,
       each detection can only be matched once
    5. Unmatched detections kept as single-modality outputs

Match threshold (12.0m) rationale:
    Standard late fusion uses 2-3m matching distance.
    We use 12m because:
    - nuScenes Mini has 0 LiDAR sweeps (single frame only)
    - PointPillars trained with 10 sweeps, runs on 1 sweep here
    - Single-sweep position estimates have ~5-10m uncertainty
    - Camera ground plane projection adds ~2-5m uncertainty
    With full 10-sweep PointPillars, threshold should be 3-4m.

Coordinate frames:
    All positions in ego vehicle frame (meters):
        x = forward  (positive = ahead of vehicle)
        y = left     (positive = left, negative = right)

LiDAR coordinate transform:
    MMDet3D outputs boxes in LiDAR sensor frame, not ego frame.
    Must apply calibrated sensor extrinsics before fusion:
        point_ego = R_lidar2ego @ point_lidar + t_lidar2ego
    See lidar_to_ego() function.

Known limitations:
    1. Single-sweep LiDAR position uncertainty (~5-10m)
       Causes some valid matches to exceed distance threshold.
       With 10-sweep PointPillars (full nuScenes), threshold
       can be reduced to 3-4m for cleaner matching.

    2. Greedy matching can consume detections suboptimally.
       Hungarian algorithm would give globally optimal matching
       but is overkill for the number of detections per frame.

    3. No temporal tracking across frames.
       Each frame is matched independently. A multi-object
       tracker (e.g. Kalman filter) would maintain object IDs
       across frames and use predicted positions for matching,
       significantly improving accuracy.

    4. Ground plane failure for occluded camera detections.
       Cars behind barriers project to wrong y position in BEV.
       These appear as camera-only detections even when LiDAR
       sees the correct position.

C++ deployment note:
    This Python implementation is the reference for the C++ port
    in deployment/src/late_fusion.cpp.

Usage:
    from late_fusion import (
        lidar_to_ego,
        deduplicate_lidar,
        fuse_detections,
        classes_compatible
    )

    # Transform LiDAR detections to ego frame
    lidar_ego = [{'class_name': c, 'score': s,
                  'bev_xy': lidar_to_ego(xy, nusc, sample)}
                 for c, s, xy in raw_lidar_dets]

    # Deduplicate (PointPillars sometimes double-detects)
    lidar_dedup = deduplicate_lidar(lidar_ego)

    # Fuse
    fused = fuse_detections(camera_dets, lidar_dedup)
"""

import numpy as np
from pyquaternion import Quaternion


# ── Constants ─────────────────────────────────────────────────────────────────

# PointPillars class names (MMDet3D output order)
POINTPILLARS_CLASS_NAMES = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

# YOLO26n class names
YOLO_CLASS_NAMES = [
    'car', 'pedestrian', 'bicycle', 'motorcycle',
    'bus', 'truck', 'traffic_cone', 'barrier'
]

# Fusion parameters
MATCH_DIST_THRESH    = 12.0   # meters — max distance for valid match
                               # larger than typical due to single-sweep LiDAR
CLASS_PENALTY        = 5.0    # meters — penalty for cross-class matches
                               # e.g. truck<->car adds 5m to score
DEDUP_DIST_THRESH    = 2.0    # meters — max distance to consider duplicate
MAX_LIDAR_DIST       = 50.0   # meters — PointPillars trained range
MAX_CAMERA_DIST      = 60.0   # meters — camera projection reliable range


# ── Coordinate transform ──────────────────────────────────────────────────────

def lidar_to_ego(bev_xy, nusc, sample):
    """
    Transform a 2D BEV position from LiDAR sensor frame to ego frame.

    MMDet3D PointPillars outputs 3D boxes in the LiDAR sensor frame
    (origin = LiDAR sensor position). Camera BEV projection outputs
    positions in the ego vehicle frame (origin = vehicle center).
    Both must be in the same frame before distance-based matching.

    Transform: point_ego = R_lidar2ego @ point_lidar + t_lidar2ego
    Where R and t come from the nuScenes calibrated_sensor record.

    Args:
        bev_xy : np.ndarray [x, y] in LiDAR sensor frame (meters)
        nusc   : NuScenes instance
        sample : nuScenes sample record

    Returns:
        np.ndarray [x, y] in ego vehicle frame (meters)
    """
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    cs         = nusc.get('calibrated_sensor',
                          lidar_data['calibrated_sensor_token'])

    R = Quaternion(cs['rotation']).rotation_matrix
    t = np.array(cs['translation'])

    pt_lidar = np.array([bev_xy[0], bev_xy[1], 0.0])
    pt_ego   = R @ pt_lidar + t

    return pt_ego[:2]


# ── Preprocessing ─────────────────────────────────────────────────────────────

def deduplicate_lidar(lidar_dets, dist_thresh=DEDUP_DIST_THRESH):
    """
    Remove duplicate LiDAR detections at nearly identical positions.

    PointPillars sometimes detects the same physical object as multiple
    class labels (e.g. bus + truck at identical coordinates). This
    happens when the 3D shape is ambiguous between similar classes.
    Keep the highest confidence detection per spatial cluster.

    Args:
        lidar_dets  : list of LiDAR detection dicts
        dist_thresh : max distance to consider two detections as
                      the same object (meters)

    Returns:
        list of deduplicated detection dicts
    """
    kept   = []
    used   = set()
    sorted_dets = sorted(lidar_dets, key=lambda x: x['score'], reverse=True)

    for i, d in enumerate(sorted_dets):
        if i in used:
            continue
        kept.append(d)
        for j, d2 in enumerate(sorted_dets):
            if j <= i or j in used:
                continue
            dist = float(np.linalg.norm(d['bev_xy'] - d2['bev_xy']))
            if dist < dist_thresh:
                used.add(j)

    return kept


def filter_lidar_front(lidar_dets, min_x=-5.0):
    """
    Filter LiDAR detections to front hemisphere only.

    Without LiDAR sweep data, PointPillars sometimes places detections
    behind the ego vehicle due to position uncertainty. Camera only
    covers the front view, so rear detections cannot be matched.
    Filter out detections behind ego (x < min_x).

    Args:
        lidar_dets : list of LiDAR detection dicts
        min_x      : minimum forward distance (meters)

    Returns:
        filtered list of detection dicts
    """
    return [d for d in lidar_dets if d['bev_xy'][0] > min_x]


# ── Class compatibility ───────────────────────────────────────────────────────

def classes_compatible(lidar_cls, camera_cls):
    """
    Check if two class names are compatible for fusion matching.

    Camera and LiDAR models may classify the same object differently
    (e.g. LiDAR sees 'bus' based on 3D dimensions, camera sees 'truck'
    based on visual appearance). Allow matching between semantically
    similar classes within the same category group.

    Args:
        lidar_cls  : str, class name from PointPillars
        camera_cls : str, class name from YOLO26n

    Returns:
        bool, True if the classes can be matched
    """
    if lidar_cls == camera_cls:
        return True

    # Large vehicles — 3D shape ambiguity between these is common
    vehicle_classes = {'car', 'truck', 'bus', 'trailer',
                       'construction_vehicle'}
    if lidar_cls in vehicle_classes and camera_cls in vehicle_classes:
        return True

    # Two-wheeled vehicles
    two_wheel_classes = {'bicycle', 'motorcycle'}
    if lidar_cls in two_wheel_classes and camera_cls in two_wheel_classes:
        return True

    return False


def is_same_large_vehicle(lidar_cls, camera_cls):
    """Check if both classes are large vehicles (truck/bus group)."""
    large = {'truck', 'bus', 'trailer', 'construction_vehicle'}
    return lidar_cls in large and camera_cls in large


# ── Core fusion ───────────────────────────────────────────────────────────────

def fuse_detections(camera_dets, lidar_dets,
                    match_thresh=MATCH_DIST_THRESH,
                    class_penalty=CLASS_PENALTY):
    """
    Fuse camera and LiDAR detections using class-aware optimal matching.

    Algorithm:
        1. Build all valid candidate pairs within match_thresh distance
           and compatible class labels
        2. Add class penalty to cross-class matches to prefer same-class
        3. Sort by penalized distance — closest/same-class matches first
        4. Greedy assignment — each detection matched at most once
        5. Output fused + unmatched single-modality detections

    For fused detections:
        - Position: LiDAR position (direct 3D measurement, more accurate)
        - Score: 0.6 * lidar_score + 0.4 * camera_score (weighted)
        - Class: LiDAR class name (3D shape more reliable for size-based
                 classification like truck vs bus)
        - bbox_px: Camera bounding box (for image overlay visualization)

    Args:
        camera_dets  : list of camera detection dicts
                       (from camera_to_bev.run_camera_to_bev)
        lidar_dets   : list of LiDAR detection dicts
                       (PointPillars output, transformed to ego frame)
        match_thresh : maximum BEV distance for valid match (meters)
        class_penalty: extra distance penalty for cross-class matches (m)

    Returns:
        list of fused detection dicts, each with:
            class_name  : str
            score       : float, combined confidence
            bev_xy      : np.ndarray [x, y] in ego frame (meters)
            source      : 'fused' | 'lidar' | 'camera'
            bbox_px     : camera bbox if available (fused/camera only)
            box_3d      : 3D box parameters if available (fused/lidar)
            match_dist  : float, matched pair distance (fused only)
            lidar_score : float (fused only)
            camera_score: float (fused only)
    """
    # Build all valid candidate pairs with penalized score
    candidates = []
    for li, ldet in enumerate(lidar_dets):
        for ci, cdet in enumerate(camera_dets):
            if not classes_compatible(ldet['class_name'],
                                      cdet['class_name']):
                continue

            dist = float(np.linalg.norm(
                ldet['bev_xy'] - cdet['bev_xy']
            ))
            if dist > match_thresh:
                continue

            # Same class or same vehicle group -> no penalty
            same_class  = ldet['class_name'] == cdet['class_name']
            same_group  = is_same_large_vehicle(ldet['class_name'],
                                                cdet['class_name'])
            penalty     = 0.0 if (same_class or same_group) else class_penalty
            score       = dist + penalty

            candidates.append((score, dist, li, ci))

    # Sort by penalized score — best matches first
    candidates.sort(key=lambda x: x[0])

    matched_cam   = set()
    matched_lidar = set()
    fused         = []

    # Greedy assignment
    for score, dist, li, ci in candidates:
        if li in matched_lidar or ci in matched_cam:
            continue

        ldet = lidar_dets[li]
        cdet = camera_dets[ci]

        fused.append({
            'class_name':   ldet['class_name'],
            'score':        float(0.6 * ldet['score'] +
                                  0.4 * cdet['score']),
            'bev_xy':       ldet['bev_xy'],   # trust LiDAR position
            'box_3d':       ldet.get('box_3d'),
            'bbox_px':      cdet.get('bbox_px'),
            'source':       'fused',
            'match_dist':   dist,
            'lidar_score':  ldet['score'],
            'camera_score': cdet['score'],
        })
        matched_lidar.add(li)
        matched_cam.add(ci)

    # Unmatched LiDAR detections
    for li, ldet in enumerate(lidar_dets):
        if li not in matched_lidar:
            d = dict(ldet)
            d['source'] = 'lidar'
            fused.append(d)

    # Unmatched camera detections
    for ci, cdet in enumerate(camera_dets):
        if ci not in matched_cam:
            d = dict(cdet)
            d['source'] = 'camera'
            fused.append(d)

    return fused


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    """
    Demo: run fusion on nuScenes sample 1 using real PointPillars output.

    Requires:
        - camera_to_bev.py for camera detections
        - PointPillars detections (from MMDet3D or NVIDIA CUDA-PointPillars)
        - nuScenes Mini dataset at /data/sets/nuscenes
    """
    import argparse
    from nuscenes.nuscenes import NuScenes
    from ultralytics import YOLO
    from camera_to_bev import run_camera_to_bev

    parser = argparse.ArgumentParser(
        description='Camera-LiDAR late fusion demo'
    )
    parser.add_argument('--nuscenes_root', default='/data/sets/nuscenes')
    parser.add_argument('--weights',
                        default='/content/drive/MyDrive/yolo_runs/'
                                'yolo26n_nuscenes/weights/best.pt')
    parser.add_argument('--sample_idx',   type=int,   default=1)
    parser.add_argument('--score_thresh', type=float, default=0.3)
    args = parser.parse_args()

    nusc   = NuScenes(version='v1.0-mini',
                      dataroot=args.nuscenes_root, verbose=False)
    model  = YOLO(args.weights)
    sample = nusc.sample[args.sample_idx]

    # Camera detections
    camera_dets, _ = run_camera_to_bev(
        nusc, model,
        sample_idx=args.sample_idx,
        score_thresh=args.score_thresh
    )
    camera_dets = [d for d in camera_dets
                   if d['distance'] <= MAX_CAMERA_DIST]

    print(f"Camera detections: {len(camera_dets)}")
    print("Note: LiDAR detections require PointPillars inference.")
    print("      See fusion notebook for complete pipeline.")
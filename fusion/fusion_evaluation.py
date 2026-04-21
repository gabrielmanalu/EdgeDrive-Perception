"""
Camera-LiDAR Fusion Evaluation
================================
Evaluates and compares detection performance across three modalities:
camera-only, LiDAR-only, and fused — using nuScenes ground truth
annotations as reference.

What this measures:
    For each detection modality we compute:
    - Precision  : of all detections, how many are correct?
    - Recall     : of all GT objects, how many were detected?
    - F1 score   : harmonic mean of precision and recall

    A detection is considered correct (true positive) if:
    1. BEV distance to nearest GT box < match_thresh (default 2.0m)
    2. Class label matches GT class (compatible classes allowed)

Evaluation approach (BEV distance matching):
    nuScenes official evaluation uses complex IoU-based 3D matching.
    We use simpler BEV distance matching because:
    - Camera detections have no 3D box (only BEV position)
    - Single-sweep LiDAR gives imprecise box dimensions
    - Distance matching is interpretable and sufficient for
      comparing relative performance across modalities

Ground truth handling:
    nuScenes annotations are in global frame.
    We transform GT boxes to ego frame for BEV comparison:
        point_ego = R_ego2global_inv @ (point_global - t_ego2global)

Key results (nuScenes Mini, sample 1, thresh=2.0m):
    Modality    Precision  Recall    F1
    Camera      high       medium    limited by ground plane errors
    LiDAR       medium     medium    limited by single-sweep uncertainty
    Fused       higher     higher    complementary strengths combine

Note on single-sweep LiDAR:
    nuScenes Mini has 0 LiDAR sweeps. PointPillars needs 10 sweeps.
    Evaluation on Mini shows degraded LiDAR performance.
    Full nuScenes (10 sweeps) would show LiDAR significantly
    outperforming camera on precision.

Usage:
    python fusion_evaluation.py \
        --nuscenes_root /data/sets/nuscenes \
        --sample_idx 1
"""

import os
import numpy as np
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes


# ── nuScenes class groupings for evaluation ───────────────────────────────────

# Map nuScenes fine-grained categories to our 8 detection classes
NUSCENES_TO_EVAL_CLASS = {
    'vehicle.car':                  'car',
    'vehicle.truck':                'truck',
    'vehicle.bus.bendy':            'bus',
    'vehicle.bus.rigid':            'bus',
    'vehicle.motorcycle':           'motorcycle',
    'vehicle.bicycle':              'bicycle',
    'vehicle.trailer':              'truck',       # grouped with truck
    'vehicle.construction':         'truck',       # grouped with truck
    'human.pedestrian.adult':       'pedestrian',
    'human.pedestrian.child':       'pedestrian',
    'human.pedestrian.wheelchair':  'pedestrian',
    'human.pedestrian.stroller':    'pedestrian',
    'movable_object.trafficcone':   'traffic_cone',
    'movable_object.barrier':       'barrier',
}

EVAL_CLASSES = [
    'car', 'pedestrian', 'bicycle', 'motorcycle',
    'bus', 'truck', 'traffic_cone', 'barrier'
]


# ── Ground truth loading ──────────────────────────────────────────────────────

def get_gt_boxes_ego(nusc, sample):
    """
    Load ground truth 3D boxes for a sample, transformed to ego frame.

    nuScenes GT annotations are stored in global frame.
    Transform: point_ego = R_inv @ (point_global - t_ego)

    Args:
        nusc   : NuScenes instance
        sample : nuScenes sample record

    Returns:
        list of dicts with keys:
            class_name : str (mapped to our 8 classes)
            bev_xy     : np.ndarray [x, y] in ego frame (meters)
            distance   : float, distance from ego (meters)
    """
    # Get ego pose for this sample
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego_pose   = nusc.get('ego_pose', lidar_data['ego_pose_token'])

    ego_translation = np.array(ego_pose['translation'])
    ego_rotation    = Quaternion(ego_pose['rotation'])

    gt_boxes = []
    for ann_token in sample['anns']:
        ann      = nusc.get('sample_annotation', ann_token)
        category = ann['category_name']

        # Map to our eval classes
        cls_name = NUSCENES_TO_EVAL_CLASS.get(category)
        if cls_name is None:
            continue  # skip unknown categories

        # Transform global -> ego frame
        pos_global = np.array(ann['translation'])
        pos_ego    = ego_rotation.inverse.rotate(
            pos_global - ego_translation
        )

        distance = float(np.sqrt(pos_ego[0]**2 + pos_ego[1]**2))

        gt_boxes.append({
            'class_name': cls_name,
            'bev_xy':     pos_ego[:2],
            'distance':   distance,
            'token':      ann_token,
        })

    return gt_boxes


# ── Matching ──────────────────────────────────────────────────────────────────

def match_detections_to_gt(detections, gt_boxes,
                            match_thresh=2.0,
                            max_dist=50.0):
    """
    Match predicted detections to GT boxes using BEV distance.

    A detection is a true positive if:
    1. BEV distance to nearest unmatched GT box < match_thresh
    2. Class labels are compatible (same class or same vehicle group)

    Args:
        detections  : list of detection dicts (bev_xy, class_name)
        gt_boxes    : list of GT box dicts (bev_xy, class_name)
        match_thresh: maximum BEV distance for TP (meters)
        max_dist    : only evaluate GT boxes within this range

    Returns:
        tp          : int, true positives
        fp          : int, false positives
        fn          : int, false negatives (missed GT boxes)
        matched_gt  : set of matched GT indices
    """
    from late_fusion import classes_compatible

    # Filter GT to evaluation range
    gt_in_range = [g for g in gt_boxes if g['distance'] <= max_dist]

    matched_gt   = set()
    matched_pred = set()
    tp = 0

    # Sort detections by confidence for greedy matching
    sorted_dets = sorted(enumerate(detections),
                         key=lambda x: x[1].get('score', 0),
                         reverse=True)

    for pred_idx, det in sorted_dets:
        best_dist = match_thresh
        best_gi   = None

        for gi, gt in enumerate(gt_in_range):
            if gi in matched_gt:
                continue
            if not classes_compatible(det['class_name'],
                                      gt['class_name']):
                continue
            dist = float(np.linalg.norm(
                det['bev_xy'] - gt['bev_xy']
            ))
            if dist < best_dist:
                best_dist = dist
                best_gi   = gi

        if best_gi is not None:
            tp += 1
            matched_gt.add(best_gi)
            matched_pred.add(pred_idx)

    fp = len(detections) - tp
    fn = len(gt_in_range) - tp

    return tp, fp, fn, matched_gt


def compute_metrics(tp, fp, fn):
    """Compute precision, recall, F1 from TP/FP/FN counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0.0)
    return precision, recall, f1


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate_fusion(nusc, camera_dets, lidar_dets, fused_dets,
                    sample, match_thresh=2.0, max_dist=50.0):
    """
    Compare camera, LiDAR, and fused detection performance
    against nuScenes ground truth annotations.

    Args:
        nusc        : NuScenes instance
        camera_dets : list of camera-only detection dicts
        lidar_dets  : list of LiDAR-only detection dicts
        fused_dets  : list of fused detection dicts (from late_fusion)
        sample      : nuScenes sample record
        match_thresh: BEV distance threshold for TP matching (meters)
        max_dist    : evaluation range (meters)

    Returns:
        dict with evaluation results per modality and per class
    """
    gt_boxes = get_gt_boxes_ego(nusc, sample)
    gt_in_range = [g for g in gt_boxes if g['distance'] <= max_dist]

    print(f"\nGround truth boxes (within {max_dist}m): {len(gt_in_range)}")
    print(f"GT class distribution:")
    for cls in EVAL_CLASSES:
        count = sum(1 for g in gt_in_range if g['class_name'] == cls)
        if count > 0:
            print(f"  {cls:<15} {count}")

    results = {}

    modalities = {
        'camera': camera_dets,
        'lidar':  lidar_dets,
        'fused':  fused_dets,
    }

    print(f"\n{'='*60}")
    print(f"{'Modality':<10} {'Dets':>5} {'TP':>5} {'FP':>5} "
          f"{'FN':>5} {'Prec':>7} {'Recall':>7} {'F1':>7}")
    print(f"{'='*60}")

    for name, dets in modalities.items():
        # Filter to evaluation range
        dets_filtered = [d for d in dets
                         if float(np.linalg.norm(d['bev_xy'])) <= max_dist]

        tp, fp, fn, _ = match_detections_to_gt(
            dets_filtered, gt_in_range, match_thresh, max_dist
        )
        precision, recall, f1 = compute_metrics(tp, fp, fn)

        results[name] = {
            'n_dets':    len(dets_filtered),
            'tp':        tp,
            'fp':        fp,
            'fn':        fn,
            'precision': precision,
            'recall':    recall,
            'f1':        f1,
        }

        print(f"{name:<10} {len(dets_filtered):>5} {tp:>5} {fp:>5} "
              f"{fn:>5} {precision:>7.3f} {recall:>7.3f} {f1:>7.3f}")

    print(f"{'='*60}")

    # Fusion improvement summary
    cam_f1    = results['camera']['f1']
    lidar_f1  = results['lidar']['f1']
    fused_f1  = results['fused']['f1']
    best_solo = max(cam_f1, lidar_f1)

    print(f"\nFusion improvement over best single modality:")
    print(f"  Best single : {best_solo:.3f} "
          f"({'camera' if cam_f1 > lidar_f1 else 'lidar'})")
    print(f"  Fused       : {fused_f1:.3f}")
    if best_solo > 0:
        improvement = (fused_f1 - best_solo) / best_solo * 100
        print(f"  Improvement : {improvement:+.1f}%")

    return results


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    """
    Run evaluation on nuScenes Mini sample 1.
    Requires PointPillars detections from MMDet3D notebook.
    """
    import argparse
    parser = argparse.ArgumentParser(
        description='Evaluate camera vs LiDAR vs fused detection'
    )
    parser.add_argument('--nuscenes_root', default='/data/sets/nuscenes')
    parser.add_argument('--sample_idx',   type=int,   default=1)
    parser.add_argument('--match_thresh', type=float, default=2.0)
    parser.add_argument('--max_dist',     type=float, default=50.0)
    args = parser.parse_args()

    nusc   = NuScenes(version='v1.0-mini',
                      dataroot=args.nuscenes_root, verbose=False)
    sample = nusc.sample[args.sample_idx]

    gt_boxes = get_gt_boxes_ego(nusc, sample)
    print(f"Sample {args.sample_idx}: {len(gt_boxes)} GT annotations")
    print("Note: Full evaluation requires camera + LiDAR detections.")
    print("      See fusion notebook for complete pipeline with results.")
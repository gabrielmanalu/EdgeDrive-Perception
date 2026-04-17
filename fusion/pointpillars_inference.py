"""
PointPillars Inference & nuScenes Evaluation
=============================================
Runs PointPillars inference on nuScenes LiDAR data and evaluates
against official nuScenes metrics (mAP, NDS).

Two inference modes:
    1. Single-frame  — one LiDAR frame, fast, for quick testing
    2. Full val set  — all 81 mini_val samples, for official evaluation

Coordinate transform pipeline:
    MMDetection3D outputs boxes in the EGO vehicle frame.
    nuScenes evaluation expects boxes in the GLOBAL frame.
    Transform: ego frame → global frame using full quaternion rotation.

    Why quaternion instead of yaw-only:
        Simple yaw extraction (atan2 from quaternion) loses pitch/roll
        and produces wrong global positions. Full quaternion rotation
        via pyquaternion correctly handles all 6 DOF.

    Transform steps:
        pos_global = ego_rotation.rotate(pos_ego) + ego_translation
        yaw_global = yaw_ego + ego_rotation.yaw_pitch_roll[0]
        rotation_quat = Quaternion(axis=[0,0,1], angle=yaw_global)

Evaluation limitation — nuScenes Mini sweep data:
    nuScenes Mini does NOT include LiDAR sweep data (0 previous sweeps).
    PointPillars was trained with 10 fused sweeps per inference.
    Running on single frames produces near-zero mAP (~0.0002).

    Published performance on full nuScenes val (with 10 sweeps):
        mAP : 0.354
        NDS : 0.476

    Use --eval flag only to understand the evaluation pipeline.
    Use published numbers for portfolio reporting.

Usage:
    # Single frame inference + BEV visualization
    python pointpillars_inference.py \\
        --mode single \\
        --nuscenes_root /data/sets/nuscenes \\
        --mmdet3d_dir   ./mmdetection3d \\
        --checkpoint    ./pointpillars_weights/pointpillars_nuscenes.pth \\
        --output_dir    ./outputs

    # Full val set inference + nuScenes evaluation
    python pointpillars_inference.py \\
        --mode eval \\
        --nuscenes_root /data/sets/nuscenes \\
        --mmdet3d_dir   ./mmdetection3d \\
        --checkpoint    ./pointpillars_weights/pointpillars_nuscenes.pth \\
        --output_dir    ./outputs
"""

import os
import sys
import json
import math
import argparse
import numpy as np
from pathlib import Path
from pyquaternion import Quaternion


# ── Constants ─────────────────────────────────────────────────────────────────

CLASS_NAMES = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

SCORE_THRESH_INFERENCE = 0.3   # for BEV visualization
SCORE_THRESH_EVAL      = 0.05  # for evaluation (lower = more recall)


# ── JSON serialization ────────────────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy scalar and array types."""
    def default(self, obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ── Coordinate transform ──────────────────────────────────────────────────────

def ego_to_global(box_xyz, box_yaw, ego_pose):
    """
    Transform a 3D box from ego vehicle frame to global frame.

    MMDetection3D outputs boxes in ego frame (origin = ego vehicle).
    nuScenes evaluation requires global frame (origin = map origin).

    Args:
        box_xyz  : np.ndarray [x, y, z] in ego frame (meters)
        box_yaw  : float, heading angle in ego frame (radians)
        ego_pose : nuScenes ego_pose record with 'translation' and 'rotation'

    Returns:
        pos_global : np.ndarray [x, y, z] in global frame
        rot_quat   : Quaternion representing global heading
    """
    ego_translation = np.array(ego_pose['translation'])
    ego_rotation    = Quaternion(ego_pose['rotation'])

    # Rotate position from ego → global, then translate
    pos_global = ego_rotation.rotate(box_xyz) + ego_translation

    # Add ego yaw to box yaw for global heading
    ego_yaw    = ego_rotation.yaw_pitch_roll[0]
    global_yaw = box_yaw + ego_yaw
    rot_quat   = Quaternion(axis=[0, 0, 1], angle=global_yaw)

    return pos_global, rot_quat


# ── Single-frame inference ────────────────────────────────────────────────────

def run_single_inference(model, nusc, nuscenes_root,
                         sample_idx=0, score_thresh=SCORE_THRESH_INFERENCE):
    """
    Run PointPillars on a single nuScenes LiDAR frame.

    Note: Single-frame inference (no sweep fusion) produces lower
    detection quality than the published mAP which uses 10 sweeps.
    nuScenes Mini does not include sweep data.

    Args:
        model         : loaded MMDet3D model
        nusc          : NuScenes instance
        nuscenes_root : path to nuScenes dataset root
        sample_idx    : which sample to run on (default: 0)
        score_thresh  : confidence threshold for filtering

    Returns:
        boxes_np  : (N, 7+) filtered box parameters
        scores_np : (N,) confidence scores
        labels_np : (N,) class label indices
        lidar_path: path to the LiDAR file used
    """
    from mmdet3d.apis import inference_detector

    sample     = nusc.sample[sample_idx]
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    lidar_path = os.path.join(nuscenes_root, lidar_data['filename'])

    print(f"Running inference on sample {sample_idx}: {lidar_path}")
    result, _ = inference_detector(model, lidar_path)

    boxes  = result.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
    scores = result.pred_instances_3d.scores_3d.cpu().numpy()
    labels = result.pred_instances_3d.labels_3d.cpu().numpy()

    mask      = scores > score_thresh
    boxes_np  = boxes[mask]
    scores_np = scores[mask]
    labels_np = labels[mask]

    print(f"✅ Inference done — {len(scores)} raw, "
          f"{mask.sum()} above threshold ({score_thresh})")
    print(f"\nDetection summary:")
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        count = (labels_np == cls_id).sum()
        if count > 0:
            print(f"  {cls_name:<25} {count}")

    return boxes_np, scores_np, labels_np, lidar_path


# ── Full val set inference ────────────────────────────────────────────────────

def get_val_samples(nusc):
    """
    Get official mini_val samples using nuScenes split definitions.
    Uses create_splits_scenes() to get the 2 official val scenes
    (scene-0103 and scene-0916) → 81 samples total.

    Important: do NOT hardcode scene indices (e.g. nusc.scene[8:])
    as the order may differ. Always use official split definitions.
    """
    from nuscenes.utils.splits import create_splits_scenes

    splits          = create_splits_scenes()
    mini_val_scenes = splits['mini_val']
    print(f"Official mini_val scenes: {mini_val_scenes}")

    val_samples = []
    for scene in nusc.scene:
        if scene['name'] in mini_val_scenes:
            token = scene['first_sample_token']
            while token:
                val_samples.append(nusc.get('sample', token))
                token = nusc.get('sample', token)['next']

    print(f"Official mini_val samples: {len(val_samples)}")
    return val_samples


def run_val_inference(model, nusc, nuscenes_root,
                      score_thresh=SCORE_THRESH_EVAL):
    """
    Run PointPillars on all official mini_val samples and collect
    results in nuScenes submission format for evaluation.

    Args:
        model         : loaded MMDet3D model
        nusc          : NuScenes instance
        nuscenes_root : path to nuScenes dataset root
        score_thresh  : minimum score for submission (default 0.05)

    Returns:
        results_dict : dict in nuScenes submission format
        val_samples  : list of val sample records
    """
    from mmdet3d.apis import inference_detector

    val_samples = get_val_samples(nusc)
    all_results = []

    print(f"\nRunning inference on {len(val_samples)} val samples...")

    for i, sample in enumerate(val_samples):
        lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        lidar_path = os.path.join(nuscenes_root, lidar_data['filename'])
        ego_pose   = nusc.get('ego_pose', lidar_data['ego_pose_token'])

        result, _ = inference_detector(model, lidar_path)

        boxes  = result.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
        scores = result.pred_instances_3d.scores_3d.cpu().numpy()
        labels = result.pred_instances_3d.labels_3d.cpu().numpy()

        for j in range(len(boxes)):
            x, y, z  = float(boxes[j][0]), float(boxes[j][1]), float(boxes[j][2])
            w, l, h  = float(boxes[j][3]), float(boxes[j][4]), float(boxes[j][5])
            yaw      = float(boxes[j][6])
            score    = float(scores[j])
            label    = int(labels[j])

            if score < score_thresh or label >= len(CLASS_NAMES):
                continue

            # Transform ego frame → global frame
            pos_global, rot_quat = ego_to_global(
                np.array([x, y, z]), yaw, ego_pose
            )

            all_results.append({
                'sample_token':    sample['token'],
                'translation':     pos_global.tolist(),
                'size':            [w, l, h],
                'rotation':        list(rot_quat),  # [w, x, y, z]
                'velocity':        [0.0, 0.0],
                'detection_name':  CLASS_NAMES[label],
                'detection_score': score,
                'attribute_name':  ''
            })

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(val_samples)} done")

    print(f"✅ Total predictions: {len(all_results)}")

    # Build nuScenes submission format
    results_dict = {
        'meta': {
            'use_lidar':    True,
            'use_camera':   False,
            'use_radar':    False,
            'use_map':      False,
            'use_external': False,
        },
        'results': {}
    }

    for r in all_results:
        token = r['sample_token']
        if token not in results_dict['results']:
            results_dict['results'][token] = []
        results_dict['results'][token].append(r)

    # Fill missing samples with empty predictions
    for sample in val_samples:
        if sample['token'] not in results_dict['results']:
            results_dict['results'][sample['token']] = []

    return results_dict, val_samples


# ── nuScenes evaluation ───────────────────────────────────────────────────────

def evaluate(nusc, results_dict, val_samples, output_dir):
    """
    Run official nuScenes detection evaluation.

    Note: Results on nuScenes Mini will be near-zero mAP because
    Mini lacks LiDAR sweep data. Use published numbers instead:
        mAP: 0.354, NDS: 0.476 (full nuScenes val, 10 sweeps)

    Args:
        nusc         : NuScenes instance
        results_dict : dict in nuScenes submission format
        val_samples  : list of val sample records
        output_dir   : directory to save evaluation results

    Returns:
        metrics : NuScenes DetectionMetrics object
    """
    from nuscenes.eval.detection.config import config_factory
    from nuscenes.eval.detection.evaluate import NuScenesEval

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results_path = os.path.join(output_dir, 'pointpillars_results.json')

    with open(results_path, 'w') as f:
        json.dump(results_dict, f, cls=NumpyEncoder)
    print(f"✅ Results saved → {results_path}")

    nusc_eval = NuScenesEval(
        nusc,
        config=config_factory('detection_cvpr_2019'),
        result_path=results_path,
        eval_set='mini_val',
        output_dir=output_dir,
        verbose=True
    )
    metrics, _ = nusc_eval.evaluate()

    print(f"\n{'='*55}")
    print(f"PointPillars — nuScenes Mini Val Evaluation")
    print(f"{'='*55}")
    print(f"mAP : {metrics.mean_ap:.4f}  "
          f"(published on full val: 0.354)")
    print(f"NDS : {metrics.nd_score:.4f}  "
          f"(published on full val: 0.476)")
    print(f"\nNote: Near-zero scores expected on Mini (no sweep data)")
    print(f"      Use published numbers for portfolio reporting")
    print(f"\nPer-class AP:")
    for cls, ap in metrics.mean_dist_aps.items():
        print(f"  {cls:<25} {ap:.4f}")

    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description='PointPillars inference on nuScenes LiDAR'
    )
    parser.add_argument('--mode', default='single',
                        choices=['single', 'eval'],
                        help='single: one frame + BEV viz | '
                             'eval: full val set + metrics')
    parser.add_argument('--nuscenes_root', default='/data/sets/nuscenes')
    parser.add_argument('--mmdet3d_dir',   default='./mmdetection3d')
    parser.add_argument('--checkpoint',
                        default='./pointpillars_weights/pointpillars_nuscenes.pth')
    parser.add_argument('--output_dir',    default='./pointpillars_outputs')
    parser.add_argument('--sample_idx',    type=int, default=0,
                        help='Sample index for single mode')
    parser.add_argument('--score_thresh',  type=float, default=0.3)
    parser.add_argument('--device',        default='cuda:0')
    return parser.parse_args()


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Setup MMDet3D
    sys.path.insert(0, args.mmdet3d_dir)
    from mmdet3d.apis import init_model
    from mmdet3d.utils import register_all_modules
    from nuscenes.nuscenes import NuScenes
    register_all_modules()

    config_path = os.path.join(
        args.mmdet3d_dir,
        'configs/pointpillars/pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py'
    )

    print(f"Loading model...")
    model = init_model(config_path, args.checkpoint, device=args.device)
    nusc  = NuScenes(version='v1.0-mini',
                     dataroot=args.nuscenes_root, verbose=False)

    if args.mode == 'single':
        # Single frame inference + BEV visualization
        boxes_np, scores_np, labels_np, lidar_path = run_single_inference(
            model, nusc, args.nuscenes_root,
            sample_idx=args.sample_idx,
            score_thresh=args.score_thresh
        )

        # Generate BEV visualization
        from bev_visualization import plot_detections_with_pointcloud
        plot_detections_with_pointcloud(
            boxes_np, scores_np, labels_np,
            lidar_path=lidar_path,
            save_path=os.path.join(args.output_dir, 'bev_detection.jpg')
        )

    elif args.mode == 'eval':
        # Full val set inference + official evaluation
        results_dict, val_samples = run_val_inference(
            model, nusc, args.nuscenes_root,
            score_thresh=args.score_thresh
        )
        evaluate(nusc, results_dict, val_samples, args.output_dir)


if __name__ == '__main__':
    main()
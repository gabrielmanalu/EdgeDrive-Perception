"""
nuScenes → YOLO Segmentation Label Converter
=============================================
Converts nuScenes 3D bounding box annotations to YOLO segmentation format
(polygon masks) for training YOLO26n-seg on front-camera driving data.

Key difference from convert_nuscenes_det.py:
    Detection    → [class cx cy w h]               (axis-aligned rectangle)
    Segmentation → [class x1 y1 x2 y2 ... xn yn]  (convex hull polygon, 4-8 points)

Why convex hull instead of rectangle:
    A car seen at an angle projects to a trapezoid, not a rectangle.
    A pedestrian projects to a narrow vertical polygon.
    Using convex hull of all 8 projected 3D box corners produces a polygon
    that follows the actual visible object silhouette — more accurate than
    a simple axis-aligned bounding box converted to a rectangle polygon.

    Example:
        Rectangle (fake):           Convex hull (real):
        ┌──────────────┐            ╱‾‾‾‾‾‾‾‾╲
        │  ░░░░░░░░░░  │           ╱  ░░░░░░░░ ╲
        │  ░░ CAR ░░  │    →     │   ░░ CAR ░░  │
        │  ░░░░░░░░░░  │           ╲  ░░░░░░░░ ╱
        └──────────────┘            ╲________╱

Label quality note:
    nuScenes does NOT provide 2D pixel-level segmentation masks —
    only 3D bounding boxes. These convex hull polygons are approximate
    silhouettes, not pixel-perfect masks. For production-quality masks,
    nuScenes-panoptic labels would be required.

Class imbalance note (nuScenes Mini):
    traffic_cone has 0 instances in the 10 mini scenes — all cone scenes
    are in the full dataset. The model will not detect cones when trained
    on mini. Full nuScenes (28k samples) resolves this.

    Class distribution (323 training images):
        car           1943  (52.2%)
        pedestrian     880  (23.7%)
        barrier        401  (10.8%)
        truck          138   (3.7%)
        bus            137   (3.7%)
        motorcycle     134   (3.6%)
        bicycle         86   (2.3%)
        traffic_cone     0   (0.0%)  ← not present in mini

Classes (8):
    0: car          4: bus
    1: pedestrian   5: truck
    2: bicycle      6: traffic_cone
    3: motorcycle   7: barrier

Usage:
    python convert_nuscenes_seg.py \\
        --nuscenes_root /data/sets/nuscenes \\
        --output_dir ./data/nuscenes_seg \\
        --version v1.0-mini

Results (nuScenes Mini):
    Total samples  : 404
    Train images   : 323
    Val images     : 81
    Total polygons : 4608
    Polygon points : 4-8 per object (convex hull of 8 projected 3D corners)
"""

import os
import argparse
import random
import shutil
import numpy as np
from pathlib import Path
from scipy.spatial import ConvexHull
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points


# ── Class mapping ─────────────────────────────────────────────────────────────

CLASS_MAP = {
    'car':          0,
    'pedestrian':   1,
    'bicycle':      2,
    'motorcycle':   3,
    'bus':          4,
    'truck':        5,
    'traffic_cone': 6,
    'barrier':      7,
}

# ── Filtering thresholds ──────────────────────────────────────────────────────

MIN_POLYGON_PTS = 4       # minimum number of valid projected corners
MIN_AREA_RATIO  = 0.0001  # polygon must cover at least 0.01% of image area
MAX_AREA_RATIO  = 0.90    # polygon must not cover more than 90% of image area
MIN_DEPTH       = 0.1     # minimum z in camera frame (meters)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert nuScenes annotations to YOLO segmentation format'
    )
    parser.add_argument('--nuscenes_root', default='/data/sets/nuscenes',
                        help='Path to nuScenes dataset root')
    parser.add_argument('--output_dir', default='./data/nuscenes_seg',
                        help='Output directory for YOLO seg labels and image symlinks')
    parser.add_argument('--version', default='v1.0-mini',
                        choices=['v1.0-mini', 'v1.0-trainval', 'v1.0-test'],
                        help='nuScenes dataset version')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Fraction of samples for validation (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible split')
    return parser.parse_args()


def collect_all_samples(nusc):
    """Collect all keyframe samples across all scenes."""
    all_samples = []
    for scene in nusc.scene:
        token = scene['first_sample_token']
        while token:
            sample = nusc.get('sample', token)
            all_samples.append(sample)
            token = sample['next']
    return all_samples


def transform_box_to_camera_frame(box, ego_pose, calibrated_sensor):
    """
    Transform a 3D box from global frame to camera frame.

    Args:
        box               : nuScenes Box object in global frame
        ego_pose          : ego_pose record for this sample
        calibrated_sensor : calibrated_sensor record for this camera

    Returns:
        box: transformed Box object in camera frame
    """
    # Step 1: Global → ego vehicle frame
    box.translate(-np.array(ego_pose['translation']))
    box.rotate(Quaternion(ego_pose['rotation']).inverse)

    # Step 2: Ego → camera frame
    box.translate(-np.array(calibrated_sensor['translation']))
    box.rotate(Quaternion(calibrated_sensor['rotation']).inverse)

    return box


def project_box_to_polygon(box, intrinsic, img_w, img_h):
    """
    Project all 8 corners of a 3D box to the image plane and compute
    their convex hull to produce a segmentation polygon.

    Unlike detection (which uses axis-aligned bbox of projected corners),
    convex hull produces a tighter polygon that follows the object silhouette.
    A car seen at an angle produces a trapezoid; a pedestrian produces a
    narrow vertical polygon.

    Args:
        box       : nuScenes Box object in camera frame
        intrinsic : 3x3 camera intrinsic matrix
        img_w     : image width in pixels
        img_h     : image height in pixels

    Returns:
        np.ndarray: shape (N, 2) normalized polygon points, or None if invalid
    """
    corners_3d = box.corners()  # (3, 8) — all 8 corners of the 3D box

    # Project all corners to image plane
    corners_2d = view_points(corners_3d, intrinsic, normalize=True)  # (3, 8)

    # Skip if any corner is behind the camera
    if any(corners_2d[2] < 0):
        return None

    xs = corners_2d[0]
    ys = corners_2d[1]

    # Keep only corners that fall within or near the image
    # Allow 10% margin outside image for partially visible objects
    valid = (
        (xs >= -img_w * 0.1) & (xs <= img_w * 1.1) &
        (ys >= -img_h * 0.1) & (ys <= img_h * 1.1)
    )

    if valid.sum() < MIN_POLYGON_PTS:
        return None

    xs_valid = xs[valid]
    ys_valid = ys[valid]

    # Compute convex hull of valid projected corners
    pts = np.stack([xs_valid, ys_valid], axis=1)
    try:
        if len(pts) >= 3:
            hull     = ConvexHull(pts)
            hull_pts = pts[hull.vertices]
        else:
            hull_pts = pts
    except Exception:
        # Degenerate case (collinear points) — fall back to bounding rectangle
        x1, x2   = xs_valid.min(), xs_valid.max()
        y1, y2   = ys_valid.min(), ys_valid.max()
        hull_pts = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])

    # Clip polygon to image bounds
    hull_pts[:, 0] = np.clip(hull_pts[:, 0], 0, img_w)
    hull_pts[:, 1] = np.clip(hull_pts[:, 1], 0, img_h)

    # Validate polygon area
    w_span     = hull_pts[:, 0].max() - hull_pts[:, 0].min()
    h_span     = hull_pts[:, 1].max() - hull_pts[:, 1].min()
    area_ratio = (w_span * h_span) / (img_w * img_h)

    if area_ratio < MIN_AREA_RATIO or area_ratio > MAX_AREA_RATIO:
        return None

    # Normalize coordinates to [0, 1]
    hull_pts[:, 0] /= img_w
    hull_pts[:, 1] /= img_h

    return hull_pts


def convert(args):
    print(f"Loading nuScenes {args.version} from {args.nuscenes_root}...")
    nusc = NuScenes(
        version=args.version,
        dataroot=args.nuscenes_root,
        verbose=False
    )

    # Fresh output directories
    output = Path(args.output_dir)
    if output.exists():
        shutil.rmtree(output)
    for split in ['train', 'val']:
        (output / 'images' / split).mkdir(parents=True, exist_ok=True)
        (output / 'labels' / split).mkdir(parents=True, exist_ok=True)

    # Collect and split samples randomly
    all_samples = collect_all_samples(nusc)
    random.seed(args.seed)
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * (1 - args.val_split))
    splits = {
        'train': all_samples[:split_idx],
        'val':   all_samples[split_idx:],
    }
    print(f"Total: {len(all_samples)} → "
          f"train: {split_idx}, val: {len(all_samples) - split_idx}")

    # Convert
    converted      = 0
    total_polygons = 0
    skipped_class  = 0
    skipped_behind = 0
    skipped_area   = 0

    for split, samples in splits.items():
        for sample in samples:
            cam_token = sample['data']['CAM_FRONT']
            cam_data  = nusc.get('sample_data', cam_token)
            img_path  = os.path.join(args.nuscenes_root, cam_data['filename'])
            img_w     = cam_data['width']
            img_h     = cam_data['height']

            cs        = nusc.get('calibrated_sensor',
                                  cam_data['calibrated_sensor_token'])
            ep        = nusc.get('ego_pose', cam_data['ego_pose_token'])
            intrinsic = np.array(cs['camera_intrinsic'])

            boxes  = nusc.get_boxes(cam_token)
            labels = []

            for box in boxes:
                # Map nuScenes category to our class index
                cat_simple = next((k for k in CLASS_MAP if k in box.name), None)
                if cat_simple is None:
                    skipped_class += 1
                    continue

                # Transform: global → ego → camera frame
                box = transform_box_to_camera_frame(box, ep, cs)

                if box.center[2] < MIN_DEPTH:
                    skipped_behind += 1
                    continue

                # Project 3D corners to convex hull polygon
                polygon = project_box_to_polygon(box, intrinsic, img_w, img_h)
                if polygon is None:
                    skipped_area += 1
                    continue

                # YOLO seg format: class x1 y1 x2 y2 x3 y3 ... xn yn
                coords = ' '.join([f"{x:.6f} {y:.6f}" for x, y in polygon])
                labels.append(f"{CLASS_MAP[cat_simple]} {coords}")
                total_polygons += 1

            # Symlink image to avoid duplicating the dataset
            img_name = Path(img_path).name
            dst_img  = output / 'images' / split / img_name
            if not dst_img.exists():
                dst_img.symlink_to(img_path)

            # Write YOLO segmentation label file
            label_path = output / 'labels' / split / f"{Path(img_name).stem}.txt"
            label_path.write_text('\n'.join(labels))

            converted += 1

    # Save dataset YAML for Ultralytics training
    yaml_content = f"""# nuScenes Segmentation Dataset
    # Generated by convert_nuscenes_seg.py
    # Version: {args.version}
    # Label format: convex hull polygons from projected 3D box corners

    path: {output.resolve()}
    train: images/train
    val: images/val

    names:
    0: car
    1: pedestrian
    2: bicycle
    3: motorcycle
    4: bus
    5: truck
    6: traffic_cone
    7: barrier
    """
    yaml_path = output / 'nuscenes_seg.yaml'
    yaml_path.write_text(yaml_content)

    # Summary report
    print(f"\n✅ Conversion complete")
    print(f"   Images converted  : {converted}")
    print(f"   Total polygons    : {total_polygons}")
    print(f"   Skipped (class)   : {skipped_class}")
    print(f"   Skipped (behind)  : {skipped_behind}")
    print(f"   Skipped (area)    : {skipped_area}")
    print(f"   YAML saved        : {yaml_path}")

    # Quality check
    print("\n--- Quality check ---")
    for split in ['train', 'val']:
        files  = list((output / 'labels' / split).glob('*.txt'))
        filled = sum(1 for f in files if f.stat().st_size > 0)
        print(f"   {split}: {len(files)} files, {filled} non-empty")

    # Spot check one polygon label
    sample_f   = next(
        f for f in (output / 'labels' / 'train').glob('*.txt')
        if f.stat().st_size > 0
    )
    first_line = open(sample_f).readline().strip().split()
    cls_id     = first_line[0]
    n_points   = (len(first_line) - 1) // 2
    print(f"\n   Sample: class={cls_id}, polygon has {n_points} points "
          f"(rectangle=4, convex hull=4-8)")
    print(f"   {' '.join(first_line[:9])} ...")


if __name__ == '__main__':
    args = parse_args()
    convert(args)
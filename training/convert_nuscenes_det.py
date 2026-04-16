"""
nuScenes → YOLO Detection Label Converter
==========================================
Converts nuScenes 3D bounding box annotations to YOLO detection format
for training YOLO26n and YOLOv8n on front-camera driving data.

Key pipeline:
    1. Load all samples across all scenes (nuScenes Mini: 404 samples)
    2. Random 80/20 split by sample — NOT by scene (see note below)
    3. For each sample, transform 3D boxes: global → ego → camera frame
    4. Project to 2D image plane using camera intrinsic matrix
    5. Filter invalid boxes (behind camera, too small, degenerate)
    6. Save normalized YOLO labels: [class cx cy w h]

Coordinate transform note:
    nuScenes stores all annotations in global world coordinates.
    Direct projection without transform produces degenerate full-image boxes.
    Correct pipeline requires 3 steps:
        Global frame
        → Ego vehicle frame  (subtract ego translation, apply ego quaternion inverse)
        → Camera frame       (subtract camera translation, apply camera quaternion inverse)
        → Image plane        (project with camera intrinsic matrix K)

Split strategy note:
    Splitting by scene (e.g. scenes 0-7 train, 8-9 val) causes empty val labels
    because some scenes have no annotations for certain classes. Random sample-level
    split ensures both splits see annotations from all scene types.

Classes (8):
    0: car          4: bus
    1: pedestrian   5: truck
    2: bicycle      6: traffic_cone
    3: motorcycle   7: barrier

Usage:
    python convert_nuscenes_det.py \
        --nuscenes_root /data/sets/nuscenes \
        --output_dir ./data/nuscenes_det \
        --version v1.0-mini

Results (nuScenes Mini):
    Total samples : 404
    Train images  : 323
    Val images    : 81
    Total boxes   : 4612
    Train labels  : 323 non-empty
    Val labels    : 80/81 non-empty
"""

import os
import argparse
import random
import shutil
import numpy as np
from pathlib import Path
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

MIN_BOX_PIXELS = 4     # minimum width or height in pixels
MAX_BOX_RATIO  = 0.95  # maximum fraction of image width/height
MIN_DEPTH      = 0.1   # minimum z in camera frame (meters)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert nuScenes annotations to YOLO detection format'
    )
    parser.add_argument('--nuscenes_root', default='/data/sets/nuscenes',
                        help='Path to nuScenes dataset root')
    parser.add_argument('--output_dir', default='./data/nuscenes_det',
                        help='Output directory for YOLO labels and image symlinks')
    parser.add_argument('--version', default='v1.0-mini',
                        choices=['v1.0-mini', 'v1.0-trainval', 'v1.0-test'],
                        help='nuScenes dataset version')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='Fraction of samples for validation (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducible split')
    return parser.parse_args()


def collect_all_samples(nusc):
    """Collect all keyframe samples across all scenes in order."""
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

    Steps:
        1. Global → ego vehicle frame
        2. Ego vehicle → camera sensor frame

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


def project_box_to_yolo(box, intrinsic, img_w, img_h):
    """
    Project a 3D box (in camera frame) to a 2D YOLO bounding box.

    Args:
        box       : nuScenes Box object in camera frame
        intrinsic : 3x3 camera intrinsic matrix
        img_w     : image width in pixels
        img_h     : image height in pixels

    Returns:
        str: YOLO label string "cx cy w h" (normalized), or None if invalid
    """
    # Skip objects behind the camera
    if box.center[2] < MIN_DEPTH:
        return None

    # Project all 8 corners of the 3D box to the image plane
    corners = view_points(box.corners(), intrinsic, normalize=True)
    xs, ys  = corners[0], corners[1]

    x1, x2 = xs.min(), xs.max()
    y1, y2  = ys.min(), ys.max()

    # Skip if completely outside image bounds
    if x2 < 0 or x1 > img_w or y2 < 0 or y1 > img_h:
        return None

    # Clip to image bounds
    x1 = np.clip(x1, 0, img_w)
    x2 = np.clip(x2, 0, img_w)
    y1 = np.clip(y1, 0, img_h)
    y2 = np.clip(y2, 0, img_h)

    # Skip tiny boxes (noise / very distant objects)
    if x2 - x1 < MIN_BOX_PIXELS or y2 - y1 < MIN_BOX_PIXELS:
        return None

    # Skip degenerate boxes covering almost the entire image
    # (caused by objects very close to camera with corners outside frame)
    if (x2 - x1) / img_w > MAX_BOX_RATIO or (y2 - y1) / img_h > MAX_BOX_RATIO:
        return None

    # Convert to YOLO normalized format: cx cy w h
    cx = (x1 + x2) / 2 / img_w
    cy = (y1 + y2) / 2 / img_h
    w  = (x2 - x1) / img_w
    h  = (y2 - y1) / img_h

    return f"{cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


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

    # Convert each sample
    converted    = 0
    total_boxes  = 0
    skipped_class   = 0
    skipped_behind  = 0
    skipped_invalid = 0

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
                # Map nuScenes category name to our class index
                cat_simple = next((k for k in CLASS_MAP if k in box.name), None)
                if cat_simple is None:
                    skipped_class += 1
                    continue

                # Transform: global → ego → camera frame
                box = transform_box_to_camera_frame(box, ep, cs)

                if box.center[2] < MIN_DEPTH:
                    skipped_behind += 1
                    continue

                # Project to 2D and validate
                yolo_box = project_box_to_yolo(box, intrinsic, img_w, img_h)
                if yolo_box is None:
                    skipped_invalid += 1
                    continue

                labels.append(f"{CLASS_MAP[cat_simple]} {yolo_box}")
                total_boxes += 1

            # Symlink image to avoid duplicating the dataset
            img_name = Path(img_path).name
            dst_img  = output / 'images' / split / img_name
            if not dst_img.exists():
                dst_img.symlink_to(img_path)

            # Write YOLO label file
            label_path = output / 'labels' / split / f"{Path(img_name).stem}.txt"
            label_path.write_text('\n'.join(labels))

            converted += 1

    # Save dataset YAML for Ultralytics training
    yaml_content = f"""# nuScenes Detection Dataset
    # Generated by convert_nuscenes_det.py
    # Version: {args.version}

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
    yaml_path = output / 'nuscenes.yaml'
    yaml_path.write_text(yaml_content)

    # Summary report
    print(f"   Conversion complete")
    print(f"   Images converted  : {converted}")
    print(f"   Total boxes       : {total_boxes}")
    print(f"   Skipped (class)   : {skipped_class}")
    print(f"   Skipped (behind)  : {skipped_behind}")
    print(f"   Skipped (invalid) : {skipped_invalid}")
    print(f"   YAML saved        : {yaml_path}")

    for split in ['train', 'val']:
        files     = list((output / 'labels' / split).glob('*.txt'))
        non_empty = sum(1 for f in files if f.stat().st_size > 0)
        print(f"\n   {split}: {len(files)} files, {non_empty} non-empty")

    # Spot check one label file
    sample_label = next(
        f for f in (output / 'labels' / 'train').glob('*.txt')
        if f.stat().st_size > 0
    )
    print(f"\n   Sample label ({sample_label.name}):")
    print('   ' + open(sample_label).read()[:200].replace('\n', '\n   '))


if __name__ == '__main__':
    args = parse_args()
    convert(args)

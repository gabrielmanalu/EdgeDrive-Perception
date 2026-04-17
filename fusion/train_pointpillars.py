"""
PointPillars Setup, Dataset Preparation & Inference Test
=========================================================
Sets up MMDetection3D, downloads pre-trained PointPillars weights,
prepares nuScenes dataset annotation files, and runs a single-frame
inference test to verify the pipeline works.

Why pre-trained weights instead of training from scratch:
    Training PointPillars on full nuScenes from scratch requires:
    - ~300GB dataset (full nuScenes with sweeps)
    - 24 epochs × ~6 hours per epoch on 8× A100 GPUs
    - Total: days of compute
    Pre-trained weights from MMDetection3D model zoo achieve
    research-grade performance (mAP 0.354, NDS 0.476 on nuScenes val)
    and are the standard approach for deployment-focused projects.

Why nuScenes Mini is insufficient for evaluation:
    nuScenes Mini (10 scenes, ~4GB) does NOT include LiDAR sweep data.
    PointPillars requires 10 fused sweeps per inference for accurate
    detection. Mini only provides single keyframes (0 previous sweeps),
    resulting in artificially low mAP (~0.0002 vs published 0.354).
    Full nuScenes (300GB) includes complete sweep history.

Published performance (full nuScenes val set, 6019 samples):
    mAP : 0.354
    NDS : 0.476  (nuScenes Detection Score — Tier IV's primary metric)
    Checkpoint: hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d

    Per-class AP (from MMDetection3D model zoo):
        car                  0.576
        truck                0.293
        bus                  0.355
        pedestrian           0.748  ← strong
        motorcycle           0.389
        bicycle              0.225
        traffic_cone         0.563
        barrier              0.553

ONNX/TensorRT export note:
    MMDetection3D v1.4 removed the ONNX export script (moved to mmdeploy).
    Even with mmdeploy, PointPillars ONNX export excludes voxelization
    (Stage 1) and post-processing (Stage 3) — these must be reimplemented
    in C++. For Jetson deployment, use NVIDIA CUDA-PointPillars which
    provides a complete C++ TensorRT pipeline:
    https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars

Installation (run once per Colab session):
    # Must use PyTorch 2.1.0 + CUDA 12.1 for mmcv pre-built wheels
    pip install torch==2.1.0 torchvision==0.16.0 \
        --index-url https://download.pytorch.org/whl/cu121
    pip install -U openmim
    mim install mmengine
    mim install mmcv==2.1.0
    mim install "mmdet>=3.0.0,<3.3.0"
    mim install "mmdet3d>=1.1.0"
    pip install nuscenes-devkit open3d

    Verified working environment:
        PyTorch : 2.1.0+cu121
        mmcv    : 2.1.0
        mmdet   : 3.2.0
        mmdet3d : 1.4.0
        mmengine: 0.10.7

Usage:
    python train_pointpillars.py \
        --nuscenes_root /data/sets/nuscenes \
        --weights_dir   ./pointpillars_weights \
        --mmdet3d_dir   ./mmdetection3d
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


# ── Constants ─────────────────────────────────────────────────────────────────

CHECKPOINT_URL = (
    'https://download.openmmlab.com/mmdetection3d/v1.0.0_models/pointpillars/'
    'hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d/'
    'hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d_20210826_104936-fca299c1.pth'
)

CONFIG_NAME = 'pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py'
SCORE_THRESH = 0.3

CLASS_NAMES = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='PointPillars setup, dataset prep and inference test'
    )
    parser.add_argument('--nuscenes_root', default='/data/sets/nuscenes',
                        help='Path to nuScenes dataset root')
    parser.add_argument('--weights_dir', default='./pointpillars_weights',
                        help='Directory to save downloaded weights')
    parser.add_argument('--mmdet3d_dir', default='./mmdetection3d',
                        help='Path to cloned MMDetection3D repo')
    parser.add_argument('--device', default='cuda:0',
                        help='Inference device')
    parser.add_argument('--skip_download', action='store_true',
                        help='Skip weight download if already exists')
    parser.add_argument('--skip_data_prep', action='store_true',
                        help='Skip pkl generation if already exists')
    return parser.parse_args()


def setup_mmdet3d(mmdet3d_dir):
    """
    Clone MMDetection3D repository if not already present.
    We need it for config files — the pip package doesn't include them.
    """
    if os.path.exists(mmdet3d_dir):
        print(f"✅ MMDetection3D already exists at {mmdet3d_dir}")
        return

    print("Cloning MMDetection3D...")
    subprocess.run([
        'git', 'clone',
        'https://github.com/open-mmlab/mmdetection3d.git',
        mmdet3d_dir, '--depth', '1'
    ], check=True)
    print(f"✅ Cloned to {mmdet3d_dir}")


def download_weights(weights_dir, skip=False):
    """
    Download pre-trained PointPillars weights from MMDetection3D model zoo.
    Checkpoint: hv_pointpillars_fpn_sbn-all_4x8_2x_nus-3d
    Published mAP: 0.354, NDS: 0.476 on full nuScenes val set.
    """
    Path(weights_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = os.path.join(weights_dir, 'pointpillars_nuscenes.pth')

    if skip and os.path.exists(ckpt_path):
        print(f"✅ Weights already exist at {ckpt_path}")
        return ckpt_path

    print("Downloading PointPillars weights (~240MB)...")
    subprocess.run([
        'wget', '-q', '--show-progress',
        '-O', ckpt_path, CHECKPOINT_URL
    ], check=True)
    print(f"✅ Weights saved to {ckpt_path}")
    return ckpt_path


def setup_nuscenes_symlink(nuscenes_root, mmdet3d_dir):
    """
    Create a symlink from mmdetection3d/data/nuscenes to your nuScenes root.
    MMDetection3D's create_data.py uses relative paths (data/nuscenes/)
    so this symlink is required for dataset preparation.
    """
    data_dir   = os.path.join(mmdet3d_dir, 'data')
    symlink    = os.path.join(data_dir, 'nuscenes')

    os.makedirs(data_dir, exist_ok=True)

    if os.path.exists(symlink):
        print(f"✅ Symlink already exists: {symlink} → {nuscenes_root}")
        return

    os.symlink(nuscenes_root, symlink)
    print(f"✅ Symlink created: {symlink} → {nuscenes_root}")


def prepare_dataset(mmdet3d_dir, skip=False):
    """
    Generate nuScenes annotation pkl files required for evaluation.
    Creates:
        nuscenes_infos_train.pkl  (323 samples)
        nuscenes_infos_val.pkl    (81 samples)
        nuscenes_dbinfos_train.pkl (ground truth database)

    Note: nuScenes Mini pkl files do NOT include LiDAR sweep data.
    This means local evaluation will show near-zero mAP because
    PointPillars needs 10 fused sweeps. Use published mAP numbers instead.
    """
    val_pkl = os.path.join(mmdet3d_dir, 'data/nuscenes/nuscenes_infos_val.pkl')

    if skip and os.path.exists(val_pkl):
        print(f"✅ Dataset pkls already exist")
        return

    print("Generating nuScenes annotation pkls (~3 min)...")
    env = os.environ.copy()
    env['PYTHONPATH'] = f"{mmdet3d_dir}:{env.get('PYTHONPATH', '')}"

    subprocess.run([
        'python', 'tools/create_data.py', 'nuscenes',
        '--root-path', 'data/nuscenes',
        '--out-dir',   'data/nuscenes',
        '--extra-tag', 'nuscenes',
        '--version',   'v1.0-mini'
    ], check=True, cwd=mmdet3d_dir, env=env)

    print(f"✅ Dataset pkls saved to {mmdet3d_dir}/data/nuscenes/")


def load_model(mmdet3d_dir, weights_dir, device):
    """
    Initialize PointPillars model with pre-trained weights.
    Adds mmdetection3d to sys.path so mmdet3d imports resolve correctly.
    """
    sys.path.insert(0, mmdet3d_dir)

    from mmdet3d.apis import init_model
    from mmdet3d.utils import register_all_modules
    register_all_modules()

    config_path = os.path.join(
        mmdet3d_dir, 'configs/pointpillars', CONFIG_NAME
    )
    ckpt_path = os.path.join(weights_dir, 'pointpillars_nuscenes.pth')

    print(f"Config    : {config_path}")
    print(f"Checkpoint: {ckpt_path}")

    model = init_model(config_path, ckpt_path, device=device)
    print(f"✅ Model loaded on {device}")
    return model, config_path


def run_inference_test(model, nuscenes_root):
    """
    Run single-frame inference test on first nuScenes sample.
    Note: Single frame only (no sweep fusion) — detection quality
    will be lower than the published mAP which uses 10 sweeps.
    """
    from mmdet3d.apis import inference_detector
    from nuscenes.nuscenes import NuScenes

    nusc       = NuScenes(version='v1.0-mini',
                          dataroot=nuscenes_root, verbose=False)
    sample     = nusc.sample[0]
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    lidar_path = os.path.join(nuscenes_root, lidar_data['filename'])

    print(f"\nRunning inference on: {lidar_path}")
    result, _ = inference_detector(model, lidar_path)

    boxes  = result.pred_instances_3d.bboxes_3d
    scores = result.pred_instances_3d.scores_3d
    labels = result.pred_instances_3d.labels_3d

    import numpy as np
    mask      = scores.cpu().numpy() > SCORE_THRESH
    boxes_np  = boxes.tensor.cpu().numpy()[mask]
    scores_np = scores.cpu().numpy()[mask]
    labels_np = labels.cpu().numpy()[mask]

    print(f"✅ Inference done")
    print(f"   Raw detections  : {len(scores)}")
    print(f"   Above threshold : {mask.sum()} (score > {SCORE_THRESH})")
    print(f"\n   Detection summary:")
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        count = (labels_np == cls_id).sum()
        if count > 0:
            print(f"     {cls_name:<25} {count}")

    return boxes_np, scores_np, labels_np


def main():
    args = parse_args()

    print("=" * 55)
    print("PointPillars Setup & Inference Test")
    print("=" * 55)

    # Step 1 — Clone MMDetection3D
    setup_mmdet3d(args.mmdet3d_dir)

    # Step 2 — Download weights
    download_weights(args.weights_dir, skip=args.skip_download)

    # Step 3 — Create nuScenes symlink
    setup_nuscenes_symlink(args.nuscenes_root, args.mmdet3d_dir)

    # Step 4 — Generate dataset pkls
    prepare_dataset(args.mmdet3d_dir, skip=args.skip_data_prep)

    # Step 5 — Load model
    model, config_path = load_model(
        args.mmdet3d_dir, args.weights_dir, args.device
    )

    # Step 6 — Run inference test
    boxes_np, scores_np, labels_np = run_inference_test(
        model, args.nuscenes_root
    )

    print(f"\n{'=' * 55}")
    print(f"✅ Setup complete. Ready for inference and BEV visualization.")
    print(f"   See pointpillars_inference.py for full pipeline.")
    print(f"{'=' * 55}")


if __name__ == '__main__':
    main()
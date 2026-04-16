"""
Speed Estimation Demo — nuScenes CAM_FRONT
==========================================
Runs Ultralytics SpeedEstimator on a nuScenes front-camera video and
saves an annotated output video with per-object speed overlays.

Important limitation — moving camera:
    SpeedEstimator works by measuring pixel displacement of tracked objects
    between frames and converting to real-world speed using meter_per_pixel.
    It assumes a STATIONARY camera (fixed CCTV/intersection camera).

    On a moving dashcam platform like nuScenes:
        - Stationary objects (signs, buildings) appear to move
        - Object pixel displacement includes ego vehicle motion
        - Resulting speeds are incorrect for most frames

    This script produces accurate results ONLY for frames where the
    ego vehicle is near-stationary (stopped at traffic lights, etc.).

meter_per_pixel calibration:
    The conversion factor depends on camera focal length and object depth:

        meter_per_pixel = object_depth_meters / focal_length_pixels

    nuScenes CAM_FRONT intrinsics:
        Focal length: 1266.4 px (both fx and fy)

    Calibrated values by assumed object depth:
        depth=5m  → meter_per_pixel=0.0039
        depth=10m → meter_per_pixel=0.0079
        depth=15m → meter_per_pixel=0.0118
        depth=20m → meter_per_pixel=0.0158

    Default used: 0.0040 (calibrated for objects at ~5m, ego stopped at
    traffic light). A walking pedestrian should read ~4-6 km/h.

    The correct production solution is Camera-LiDAR fusion:
    LiDAR provides per-object depth, enabling accurate speed estimation
    regardless of object distance or camera motion. See fusion/ module.

Sanity check values:
    Walking pedestrian : 4–6 km/h
    Slow moving car    : 10–30 km/h
    Stationary object  : 0–2 km/h

Usage:
    python speed_estimation_demo.py \\
        --model ./runs/yolo26n_nuscenes/weights/best.pt \\
        --video ./videos/scene_02_scene-0553.mp4 \\
        --output ./solutions_output \\
        --meter_per_pixel 0.0040
"""

import argparse
import os
import cv2
from pathlib import Path
from ultralytics import solutions


def parse_args():
    parser = argparse.ArgumentParser(
        description='Speed estimation demo on nuScenes video'
    )
    parser.add_argument('--model',  required=True,
                        help='Path to trained YOLO26n best.pt')
    parser.add_argument('--video',  required=True,
                        help='Path to input video (from export_nuscenes_video.py)')
    parser.add_argument('--output', default='./solutions_output',
                        help='Output directory for annotated video')
    parser.add_argument('--meter_per_pixel', type=float, default=0.0040,
                        help='Calibrated m/px for nuScenes CAM_FRONT at ~5m depth')
    return parser.parse_args()


def main():
    args = parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(args.output, 'speed.mp4')

    # Get video properties
    cap = cv2.VideoCapture(args.video)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 12
    cap.release()

    print(f"Input : {args.video}")
    print(f"Size  : {W}x{H} @ {FPS:.1f} FPS")
    print(f"meter_per_pixel : {args.meter_per_pixel}")
    print(f"Output: {out_path}")
    print()

    speed_obj = solutions.SpeedEstimator(
        model=args.model,
        show=False,
        meter_per_pixel=args.meter_per_pixel,
    )

    cap = cv2.VideoCapture(args.video)
    out = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        FPS, (W, H)
    )

    frame_count = 0
    print("Running Speed Estimator...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(speed_obj.process(frame).plot_im)
        frame_count += 1

    cap.release()
    out.release()

    size_kb = Path(out_path).stat().st_size / 1024
    print(f"✅ Done — {frame_count} frames → {out_path} ({size_kb:.0f} KB)")
    print()
    print("NOTE: Accuracy depends on ego vehicle being near-stationary.")
    print("      For moving camera, use Camera-LiDAR fusion for true speed.")


if __name__ == '__main__':
    main()
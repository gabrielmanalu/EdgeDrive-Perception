"""
Traffic Density Heatmap Demo — nuScenes CAM_FRONT
==================================================
Runs Ultralytics Heatmap on a nuScenes front-camera video, accumulating
detection centroids over time to produce a traffic density heatmap.

How it works:
    For each frame, detected object centroids are added to a persistent
    float heatmap buffer as Gaussian blobs. The buffer accumulates across
    all frames — areas where objects appear frequently become hotter.
    A colormap is applied for visualization (PARULA: blue=sparse, yellow=dense).

    This produces a visual summary of WHERE objects tend to appear in
    a driving scene over time — useful for understanding traffic patterns,
    identifying high-density intersection zones, and validating detection
    coverage across the camera FOV.

Unlike speed estimation, heatmap works correctly on a moving camera
because it only tracks WHERE objects appear in the image frame, not
how fast they move in the real world.

Colormap options (OpenCV):
    cv2.COLORMAP_PARULA  ← default, perceptually uniform, blue→yellow
    cv2.COLORMAP_JET     ← classic blue→red, not perceptually uniform
    cv2.COLORMAP_HOT     ← black→red→yellow→white
    cv2.COLORMAP_TURBO   ← improved rainbow, Google's alternative to JET

Usage:
    python heatmap_demo.py \\
        --model ./runs/yolo26n_nuscenes/weights/best.pt \\
        --video ./videos/scene_00_scene-0061.mp4 \\
        --output ./solutions_output \\
        --colormap PARULA
"""

import argparse
import os
import cv2
from pathlib import Path
from ultralytics import solutions

COLORMAP_MAP = {
    'PARULA': cv2.COLORMAP_PARULA,
    'JET':    cv2.COLORMAP_JET,
    'HOT':    cv2.COLORMAP_HOT,
    'TURBO':  cv2.COLORMAP_TURBO,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Traffic density heatmap demo on nuScenes video'
    )
    parser.add_argument('--model',    required=True,
                        help='Path to trained YOLO26n best.pt')
    parser.add_argument('--video',    required=True,
                        help='Path to input video (from export_nuscenes_video.py)')
    parser.add_argument('--output',   default='./solutions_output',
                        help='Output directory for annotated video')
    parser.add_argument('--colormap', default='PARULA',
                        choices=list(COLORMAP_MAP.keys()),
                        help='Heatmap colormap (default: PARULA)')
    return parser.parse_args()


def main():
    args = parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(args.output, 'heatmap.mp4')

    cap = cv2.VideoCapture(args.video)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 12
    cap.release()

    print(f"Input    : {args.video}")
    print(f"Size     : {W}x{H} @ {FPS:.1f} FPS")
    print(f"Colormap : {args.colormap}")
    print(f"Output   : {out_path}")
    print()

    heat_obj = solutions.Heatmap(
        model=args.model,
        colormap=COLORMAP_MAP[args.colormap],
        show=False,
    )

    cap = cv2.VideoCapture(args.video)
    out = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        FPS, (W, H)
    )

    frame_count = 0
    print("Running Heatmap...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(heat_obj.process(frame).plot_im)
        frame_count += 1

    cap.release()
    out.release()

    size_kb = Path(out_path).stat().st_size / 1024
    print(f"✅ Done — {frame_count} frames → {out_path} ({size_kb:.0f} KB)")


if __name__ == '__main__':
    main()
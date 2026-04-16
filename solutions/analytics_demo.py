"""
Real-Time Detection Analytics Demo — nuScenes CAM_FRONT
=======================================================
Runs Ultralytics Analytics on a nuScenes front-camera video, generating
a real-time line chart of detection counts per class over time.

How it works:
    For each frame, YOLO26n runs detection and the Analytics module
    updates a live line chart showing how many objects of each class
    are detected across the video timeline.

    This produces a temporal visualization of traffic density:
        - Spikes show frames with many objects
        - Class-specific lines show which objects dominate the scene
        - Useful for understanding scene complexity over a driving sequence

Output note:
    Analytics output frame size is fixed at 1280x720 regardless of
    input video resolution. The VideoWriter must use (1280, 720) not
    the input video dimensions — using the wrong size produces a 1KB
    empty file (silent failure from OpenCV).

    This was discovered during development: output size must be
    checked from result.plot_im.shape before initializing VideoWriter.

Analytics types:
    'line'  ← default, shows detection count over time per class
    'bar'   ← bar chart of total detections per class
    'pie'   ← pie chart of class distribution

Usage:
    python analytics_demo.py \\
        --model ./runs/yolo26n_nuscenes/weights/best.pt \\
        --video ./videos/scene_00_scene-0061.mp4 \\
        --output ./solutions_output \\
        --analytics_type line
"""

import argparse
import os
import cv2
from pathlib import Path
from ultralytics import solutions


def parse_args():
    parser = argparse.ArgumentParser(
        description='Real-time detection analytics demo on nuScenes video'
    )
    parser.add_argument('--model',          required=True,
                        help='Path to trained YOLO26n best.pt')
    parser.add_argument('--video',          required=True,
                        help='Path to input video (from export_nuscenes_video.py)')
    parser.add_argument('--output',         default='./solutions_output',
                        help='Output directory for annotated video')
    parser.add_argument('--analytics_type', default='line',
                        choices=['line', 'bar', 'pie'],
                        help='Chart type (default: line)')
    return parser.parse_args()


def main():
    args = parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(args.output, 'analytics.mp4')

    cap = cv2.VideoCapture(args.video)
    FPS = cap.get(cv2.CAP_PROP_FPS) or 12
    cap.release()

    print(f"Input          : {args.video}")
    print(f"Analytics type : {args.analytics_type}")
    print(f"Output         : {out_path}")
    print()

    analytics_obj = solutions.Analytics(
        model=args.model,
        analytics_type=args.analytics_type,
        show=False,
    )

    cap = cv2.VideoCapture(args.video)
    out         = None  # initialized after first frame (output size differs from input)
    frame_idx   = 0
    frame_count = 0

    print("Running Analytics...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        result   = analytics_obj.process(frame, frame_idx)
        plot_im  = result.plot_im

        # Initialize writer on first frame using ACTUAL output dimensions
        # Analytics output is always 1280x720 regardless of input resolution
        if out is None:
            out_h, out_w = plot_im.shape[:2]
            out = cv2.VideoWriter(
                out_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                FPS,
                (out_w, out_h)
            )
            print(f"Analytics output size: {out_w}x{out_h} "
                  f"(fixed, independent of input)")

        out.write(plot_im)
        frame_count += 1

    cap.release()
    if out:
        out.release()

    size_kb = Path(out_path).stat().st_size / 1024
    print(f"✅ Done — {frame_count} frames → {out_path} ({size_kb:.0f} KB)")


if __name__ == '__main__':
    main()
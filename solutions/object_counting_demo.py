"""
Zone-Based Object Counting Demo — nuScenes CAM_FRONT
=====================================================
Runs Ultralytics ObjectCounter on a nuScenes front-camera video,
counting objects that cross a defined region line in the frame.

How it works:
    A counting line/region is drawn across the image at a specified
    vertical position. When a tracked object's centroid crosses this
    line, it is counted as either IN (top→bottom) or OUT (bottom→top).
    Persistent track IDs ensure each object is counted only once.

    In autonomous driving context:
        - Count vehicles approaching an intersection
        - Count pedestrians crossing a marked zone
        - Monitor traffic flow direction at a chokepoint

Counting region:
    Default region is a horizontal line at 65% of frame height:
        [(0, H*0.65), (W, H*0.65)]

    This places the line in the lower-middle of the frame where
    vehicles and pedestrians are most likely to be fully visible
    and tracked with stable IDs before crossing.

    Adjust with --region_y (0.0–1.0 as fraction of frame height).

Unlike speed estimation, object counting works correctly on a moving
camera — it counts image-plane crossings, not real-world distances.

Usage:
    python object_counting_demo.py \\
        --model ./runs/yolo26n_nuscenes/weights/best.pt \\
        --video ./videos/scene_00_scene-0061.mp4 \\
        --output ./solutions_output \\
        --region_y 0.65
"""

import argparse
import os
import cv2
from pathlib import Path
from ultralytics import solutions


def parse_args():
    parser = argparse.ArgumentParser(
        description='Zone-based object counting demo on nuScenes video'
    )
    parser.add_argument('--model',    required=True,
                        help='Path to trained YOLO26n best.pt')
    parser.add_argument('--video',    required=True,
                        help='Path to input video (from export_nuscenes_video.py)')
    parser.add_argument('--output',   default='./solutions_output',
                        help='Output directory for annotated video')
    parser.add_argument('--region_y', type=float, default=0.65,
                        help='Vertical position of counting line as fraction '
                             'of frame height (default: 0.65)')
    return parser.parse_args()


def main():
    args = parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(args.output, 'counting.mp4')

    cap = cv2.VideoCapture(args.video)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 12
    cap.release()

    # Horizontal counting line at region_y fraction of frame height
    line_y = int(H * args.region_y)
    region = [(0, line_y), (W, line_y)]

    print(f"Input    : {args.video}")
    print(f"Size     : {W}x{H} @ {FPS:.1f} FPS")
    print(f"Region   : horizontal line at y={line_y}px ({args.region_y:.0%} of height)")
    print(f"Output   : {out_path}")
    print()

    count_obj = solutions.ObjectCounter(
        model=args.model,
        region=region,
        show=False,
    )

    cap = cv2.VideoCapture(args.video)
    out = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        FPS, (W, H)
    )

    frame_count = 0
    print("Running Object Counter...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result = count_obj.process(frame)
        out.write(result.plot_im)
        frame_count += 1

    cap.release()
    out.release()

    size_kb = Path(out_path).stat().st_size / 1024
    print(f"✅ Done — {frame_count} frames → {out_path} ({size_kb:.0f} KB)")
    print(f"   Final counts: IN={result.in_count}, OUT={result.out_count}")


if __name__ == '__main__':
    main()
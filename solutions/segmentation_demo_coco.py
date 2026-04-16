"""
Instance Segmentation Demo — Base COCO Weights (Pixel-Perfect Masks)
=====================================================================
Runs the base COCO-pretrained YOLO26n-seg on a nuScenes front-camera
video for high-quality pixel-level instance segmentation.

Why base COCO weights instead of fine-tuned:
    The fine-tuned YOLO26n-seg model (trained on nuScenes Mini) produces
    rectangular/trapezoidal masks because the training labels were convex
    hull polygons projected from 3D bounding boxes — not true pixel masks.
    nuScenes does not provide 2D pixel-level segmentation annotations.

    Base COCO weights were trained on 118k images with pixel-perfect
    polygon masks for 80 classes. On driving footage, cars and pedestrians
    show proper contoured masks that follow the actual object silhouette.

    Visual quality comparison:
        Fine-tuned seg  → filled rectangles / trapezoids (label artifact)
        Base COCO seg   → pixel-level contoured masks ✅

Tradeoff — class coverage:
    Base COCO weights detect 80 generic classes but are missing
    autonomous driving-specific classes:
        ✅ car, person, bicycle, motorcycle, bus, truck  (in COCO)
        ❌ traffic_cone                                  (not in COCO)
        ❌ barrier                                       (not in COCO)

    For a complete solution, nuScenes-panoptic labels would enable
    fine-tuning with both pixel-perfect masks AND full class coverage.

Use this script for:
    - Visual demo footage where mask quality matters
    - Showcasing YOLO26n-seg capability on driving scenes
    - Comparing mask quality against the fine-tuned model

Use segmentation_demo.py (fine-tuned) for:
    - Detecting traffic_cone and barrier
    - nuScenes class-specific evaluation
    - Benchmark comparisons against the detection models

Usage:
    python segmentation_demo_coco.py \\
        --video ./videos/scene_00_scene-0061.mp4 \\
        --output ./solutions_output \\
        --conf 0.25
"""

import argparse
import os
import cv2
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description='Instance segmentation using base COCO weights '
                    '(pixel-perfect masks, common classes)'
    )
    parser.add_argument('--video',  required=True,
                        help='Path to input video (from export_nuscenes_video.py)')
    parser.add_argument('--output', default='./solutions_output',
                        help='Output directory for annotated video')
    parser.add_argument('--conf',   type=float, default=0.25,
                        help='Detection confidence threshold (default: 0.25)')
    return parser.parse_args()


def main():
    args = parse_args()

    Path(args.output).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(args.output, 'segmentation_coco.mp4')

    cap = cv2.VideoCapture(args.video)
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    FPS = cap.get(cv2.CAP_PROP_FPS) or 12
    cap.release()

    print(f"Input  : {args.video}")
    print(f"Model  : yolo26n-seg.pt (base COCO weights — auto-download)")
    print(f"Size   : {W}x{H} @ {FPS:.1f} FPS")
    print(f"Output : {out_path}")
    print()
    print("Note: Detects COCO classes only — traffic_cone and barrier")
    print("      will NOT be detected. Use segmentation_demo.py for those.")
    print()

    # Base COCO weights — downloads automatically on first run (~6MB)
    model = YOLO("yolo26n-seg.pt")

    cap = cv2.VideoCapture(args.video)
    out = cv2.VideoWriter(
        out_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        FPS, (W, H)
    )

    frame_count = 0
    print("Running Segmentation (COCO)...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # plot() draws bounding boxes + filled segmentation masks
        out.write(model(frame, verbose=False, conf=args.conf)[0].plot())
        frame_count += 1

    cap.release()
    out.release()

    size_kb = Path(out_path).stat().st_size / 1024
    print(f"✅ Done — {frame_count} frames → {out_path} ({size_kb:.0f} KB)")


if __name__ == '__main__':
    main()
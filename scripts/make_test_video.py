#!/usr/bin/env python3
"""
make_test_video.py — Create a driving video from nuScenes test images

Stitches the 404 nuScenes CAM_FRONT images into a .mp4 at 12 Hz
(nuScenes native camera frequency), giving ~34 seconds of footage.

Usage:
    python3 scripts/make_test_video.py
    python3 scripts/make_test_video.py --fps 12 --out test_videos/nuscenes_clip.mp4

Output:
    test_videos/nuscenes_clip.mp4  (~34s, 1600x900, 12 FPS)
"""

import argparse
import os
import sys
import glob
import cv2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default="test_images",
                        help="Directory of nuScenes CAM_FRONT images")
    parser.add_argument("--out", default="test_videos/nuscenes_clip.mp4",
                        help="Output video path")
    parser.add_argument("--fps", type=float, default=12.0,
                        help="Output FPS (nuScenes native: 12 Hz)")
    args = parser.parse_args()

    # Gather images
    patterns = [
        os.path.join(args.images, "*.jpg"),
        os.path.join(args.images, "*.png"),
    ]
    paths = []
    for p in patterns:
        paths.extend(glob.glob(p))
    paths = sorted(paths)

    if not paths:
        print(f"Error: No images found in {args.images}/")
        print("nuScenes test images should be at test_images/*.jpg")
        sys.exit(1)

    # Read first image for dimensions
    sample = cv2.imread(paths[0])
    if sample is None:
        print(f"Error: Could not read {paths[0]}")
        sys.exit(1)
    h, w = sample.shape[:2]

    print(f"Found {len(paths)} images ({w}x{h})")
    print(f"Output: {args.out} @ {args.fps} FPS")
    print(f"Duration: ~{len(paths)/args.fps:.1f}s")

    # Create output directory
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # Write video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, args.fps, (w, h))
    if not writer.isOpened():
        print(f"Error: Could not open video writer at {args.out}")
        sys.exit(1)

    for i, path in enumerate(paths):
        frame = cv2.imread(path)
        if frame is None:
            print(f"  Skipping unreadable: {path}")
            continue
        writer.write(frame)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(paths)} frames written...")

    writer.release()
    size_mb = os.path.getsize(args.out) / 1024 / 1024
    print(f"\n✅ Saved: {args.out} ({size_mb:.1f} MB)")
    print(f"\nRun inference on it:")
    print(f"  ./scripts/test_camera.sh video-save")
    print(f"  # or directly:")
    print(f"  ./deployment/build/edgedrive \\")
    print(f"      --engine weights/yolo26n_det_int8_raw.engine \\")
    print(f"      --video {args.out} \\")
    print(f"      --save-video output/demo_annotated.mp4 \\")
    print(f"      --no-display --no-loop")

if __name__ == "__main__":
    main()
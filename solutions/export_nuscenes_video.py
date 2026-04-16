"""
nuScenes Scene → MP4 Video Exporter
=====================================
Exports nuScenes front-camera scenes as smooth MP4 videos by combining
both keyframes and sweep frames in chronological order.

Keyframes vs sweeps:
    nuScenes stores two types of camera data:

    samples/   ← keyframes only, annotated, 2Hz (~1 frame every 0.5s)
    sweeps/    ← full-rate frames between keyframes, unannotated, 12Hz

    Using only keyframes (nusc.sample) produces ~2 FPS choppy video
    because only 39-41 annotated frames exist per 20-second scene.

    This script follows the sample_data linked list (sd['next']) which
    includes BOTH keyframes and sweeps in chronological order, producing
    smooth 12Hz (~12 FPS) video that looks like real driving footage.

    Result per scene:
        Keyframes only : ~40 frames  → ~2 FPS  (choppy)
        Keyframes + sweeps : ~400 frames → 12 FPS (smooth) ✅

Output:
    One MP4 file per scene, named:
        scene_00_scene-0061.mp4
        scene_01_scene-0103.mp4
        ...

    Videos are used as input for the Solutions pipeline
    (speed estimation, heatmap, object counting, analytics).

Usage:
    python export_nuscenes_video.py \\
        --nuscenes_root /data/sets/nuscenes \\
        --output_dir ./videos \\
        --version v1.0-mini \\
        --fps 12
"""

import os
import argparse
import cv2
from pathlib import Path
from nuscenes.nuscenes import NuScenes


def parse_args():
    parser = argparse.ArgumentParser(
        description='Export nuScenes CAM_FRONT scenes as MP4 videos'
    )
    parser.add_argument('--nuscenes_root', default='/data/sets/nuscenes',
                        help='Path to nuScenes dataset root')
    parser.add_argument('--output_dir', default='./videos',
                        help='Directory to save exported MP4 files')
    parser.add_argument('--version', default='v1.0-mini',
                        choices=['v1.0-mini', 'v1.0-trainval', 'v1.0-test'],
                        help='nuScenes dataset version')
    parser.add_argument('--fps', type=int, default=12,
                        help='Output video FPS (nuScenes native ~12Hz)')
    parser.add_argument('--scene', type=int, default=None,
                        help='Export only a specific scene index (default: all)')
    return parser.parse_args()


def export_scene(nusc, scene, scene_idx, output_dir, nuscenes_root, fps):
    """
    Export a single nuScenes scene as an MP4 video.

    Traverses the sample_data linked list starting from the first
    CAM_FRONT sample_data token, collecting all frames (keyframes +
    sweeps) in chronological order before writing the video.

    Args:
        nusc         : NuScenes instance
        scene        : nuScenes scene record
        scene_idx    : integer index for output filename
        output_dir   : Path to output directory
        nuscenes_root: Path to nuScenes dataset root
        fps          : Output video frame rate

    Returns:
        tuple: (output_path, frame_count)
    """
    out_path = output_dir / f"scene_{scene_idx:02d}_{scene['name']}.mp4"

    # Get first CAM_FRONT sample_data token from the first keyframe
    first_sample = nusc.get('sample', scene['first_sample_token'])
    cam_sd_token = first_sample['data']['CAM_FRONT']

    # Traverse the sample_data linked list: keyframes + sweeps in order
    # Each sample_data record has a 'next' pointer to the next frame
    # (empty string '' when at the end of the scene)
    all_frames = []
    current_token = cam_sd_token
    while current_token:
        sd = nusc.get('sample_data', current_token)
        all_frames.append(sd)
        current_token = sd['next']

    # Write video
    writer = None
    frames_written = 0

    for sd in all_frames:
        img_path = os.path.join(nuscenes_root, sd['filename'])

        if not os.path.exists(img_path):
            continue

        frame = cv2.imread(img_path)
        if frame is None:
            continue

        # Initialize writer on first valid frame (gets dimensions from image)
        if writer is None:
            h, w = frame.shape[:2]
            writer = cv2.VideoWriter(
                str(out_path),
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (w, h)
            )

        writer.write(frame)
        frames_written += 1

    if writer:
        writer.release()

    return out_path, frames_written


def main():
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading nuScenes {args.version} from {args.nuscenes_root}...")
    nusc = NuScenes(
        version=args.version,
        dataroot=args.nuscenes_root,
        verbose=False
    )

    # Select scenes to export
    scenes = nusc.scene
    if args.scene is not None:
        if args.scene >= len(scenes):
            print(f"❌ Scene {args.scene} not found (total: {len(scenes)})")
            return
        scenes = [scenes[args.scene]]
        start_idx = args.scene
    else:
        start_idx = 0

    print(f"Exporting {len(scenes)} scene(s) at {args.fps} FPS...\n")

    total_frames = 0
    for i, scene in enumerate(scenes):
        scene_idx = start_idx + i
        out_path, frame_count = export_scene(
            nusc, scene, scene_idx,
            output_dir, args.nuscenes_root, args.fps
        )
        total_frames += frame_count
        print(f"✅ Scene {scene_idx:02d}: {frame_count} frames → {out_path.name}")

    print(f"\nAll videos saved to {output_dir}")
    print(f"Total frames exported: {total_frames}")
    print(f"\nAvailable videos:")
    for f in sorted(output_dir.glob('*.mp4')):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}  ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
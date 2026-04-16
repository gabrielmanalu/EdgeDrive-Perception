"""
Run All Solutions Pipeline — nuScenes CAM_FRONT
================================================
Runs all 5 Solutions demos sequentially on a nuScenes video and saves
annotated output videos to a single output directory.

Runs in order:
    1. speed.mp4              — SpeedEstimator (calibrated meter_per_pixel)
    2. heatmap.mp4            — Traffic density heatmap (PARULA colormap)
    3. counting.mp4           — Zone-based object counting
    4. analytics.mp4          — Real-time detection count line chart
    5. segmentation_coco.mp4  — Instance masks (base COCO weights, best quality)
    6. segmentation_nuscenes.mp4 — Instance masks (fine-tuned, nuScenes classes)

Models used:
    YOLO26n-det : Speed, Heatmap, Counting, Analytics
    YOLO26n-seg : Segmentation (fine-tuned on nuScenes)
    yolo26n-seg.pt : Segmentation (base COCO, pixel-perfect masks)

Usage:
    python run_all_solutions.py \\
        --model_det  ./runs/yolo26n_nuscenes/weights/best.pt \\
        --model_seg  ./runs/yolo26n_seg_nuscenes/weights/best.pt \\
        --video      ./videos/scene_02_scene-0553.mp4 \\
        --output     ./solutions_output

    # Run only specific solutions
    python run_all_solutions.py \\
        --model_det ./runs/yolo26n_nuscenes/weights/best.pt \\
        --video ./videos/scene_00_scene-0061.mp4 \\
        --skip speed segmentation_nuscenes
"""

import argparse
import os
import cv2
import time
from pathlib import Path
from ultralytics import YOLO, solutions


def parse_args():
    parser = argparse.ArgumentParser(
        description='Run all Solutions demos on a nuScenes video'
    )
    parser.add_argument('--model_det', required=True,
                        help='Path to YOLO26n-det best.pt')
    parser.add_argument('--model_seg', default=None,
                        help='Path to YOLO26n-seg best.pt (optional)')
    parser.add_argument('--video',     required=True,
                        help='Path to input video (from export_nuscenes_video.py)')
    parser.add_argument('--output',    default='./solutions_output',
                        help='Output directory for all annotated videos')
    parser.add_argument('--meter_per_pixel', type=float, default=0.0040,
                        help='Calibrated m/px for speed estimation')
    parser.add_argument('--region_y',  type=float, default=0.65,
                        help='Counting line position (fraction of frame height)')
    parser.add_argument('--skip',      nargs='*', default=[],
                        choices=['speed', 'heatmap', 'counting',
                                 'analytics', 'segmentation_coco',
                                 'segmentation_nuscenes'],
                        help='Solutions to skip')
    return parser.parse_args()


def get_video_props(path):
    """Return (width, height, fps) for a video file."""
    cap = cv2.VideoCapture(path)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 12
    cap.release()
    return w, h, fps


def make_writer(path, fps, w, h):
    """Create an OpenCV VideoWriter."""
    return cv2.VideoWriter(
        path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (w, h)
    )


def run_speed(args, W, H, FPS):
    print("\n[1/6] Speed Estimation...")
    out_path = os.path.join(args.output, 'speed.mp4')
    speed_obj = solutions.SpeedEstimator(
        model=args.model_det,
        show=False,
        meter_per_pixel=args.meter_per_pixel,
    )
    cap = cv2.VideoCapture(args.video)
    out = make_writer(out_path, FPS, W, H)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        out.write(speed_obj.process(frame).plot_im)
    cap.release(); out.release()
    return out_path


def run_heatmap(args, W, H, FPS):
    print("\n[2/6] Heatmap...")
    out_path = os.path.join(args.output, 'heatmap.mp4')
    heat_obj = solutions.Heatmap(
        model=args.model_det,
        colormap=cv2.COLORMAP_PARULA,
        show=False,
    )
    cap = cv2.VideoCapture(args.video)
    out = make_writer(out_path, FPS, W, H)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        out.write(heat_obj.process(frame).plot_im)
    cap.release(); out.release()
    return out_path


def run_counting(args, W, H, FPS):
    print("\n[3/6] Object Counting...")
    out_path = os.path.join(args.output, 'counting.mp4')
    line_y   = int(H * args.region_y)
    count_obj = solutions.ObjectCounter(
        model=args.model_det,
        region=[(0, line_y), (W, line_y)],
        show=False,
    )
    cap = cv2.VideoCapture(args.video)
    out = make_writer(out_path, FPS, W, H)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        out.write(count_obj.process(frame).plot_im)
    cap.release(); out.release()
    return out_path


def run_analytics(args, FPS):
    print("\n[4/6] Analytics...")
    out_path     = os.path.join(args.output, 'analytics.mp4')
    analytics_obj = solutions.Analytics(
        model=args.model_det,
        analytics_type="line",
        show=False,
    )
    cap       = cv2.VideoCapture(args.video)
    out       = None
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_idx += 1
        plot_im = analytics_obj.process(frame, frame_idx).plot_im
        if out is None:
            out_h, out_w = plot_im.shape[:2]
            out = make_writer(out_path, FPS, out_w, out_h)
        out.write(plot_im)
    cap.release()
    if out: out.release()
    return out_path


def run_segmentation(args, W, H, FPS, model_path, suffix):
    label = 'COCO' if suffix == 'coco' else 'nuScenes fine-tuned'
    print(f"\n[5-6/6] Segmentation ({label})...")
    out_path = os.path.join(args.output, f'segmentation_{suffix}.mp4')
    model    = YOLO(model_path)
    cap      = cv2.VideoCapture(args.video)
    out      = make_writer(out_path, FPS, W, H)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        out.write(model(frame, verbose=False)[0].plot())
    cap.release(); out.release()
    return out_path


def main():
    args = parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)

    W, H, FPS = get_video_props(args.video)
    print(f"Video  : {args.video}")
    print(f"Size   : {W}x{H} @ {FPS:.1f} FPS")
    print(f"Output : {args.output}")
    print(f"Skip   : {args.skip or 'none'}")

    results = {}
    t_start = time.time()

    if 'speed' not in args.skip:
        results['speed'] = run_speed(args, W, H, FPS)

    if 'heatmap' not in args.skip:
        results['heatmap'] = run_heatmap(args, W, H, FPS)

    if 'counting' not in args.skip:
        results['counting'] = run_counting(args, W, H, FPS)

    if 'analytics' not in args.skip:
        results['analytics'] = run_analytics(args, FPS)

    if 'segmentation_coco' not in args.skip:
        results['segmentation_coco'] = run_segmentation(
            args, W, H, FPS, 'yolo26n-seg.pt', 'coco'
        )

    if 'segmentation_nuscenes' not in args.skip and args.model_seg:
        results['segmentation_nuscenes'] = run_segmentation(
            args, W, H, FPS, args.model_seg, 'nuscenes'
        )

    # Summary
    elapsed = time.time() - t_start
    print(f"\n{'─'*50}")
    print(f"✅ All solutions complete in {elapsed:.0f}s")
    print(f"{'─'*50}")
    for name, path in results.items():
        size_kb = Path(path).stat().st_size / 1024
        print(f"  {name:<28} {size_kb:>8.0f} KB  →  {Path(path).name}")


if __name__ == '__main__':
    main()
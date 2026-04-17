"""
Bird's Eye View (BEV) Visualization for PointPillars 3D Detections
===================================================================
Provides three BEV visualization functions for PointPillars output:

    1. plot_detections_bev()
       Basic BEV with 3D boxes on black background.
       Fast, clean, good for quick inspection.

    2. plot_pointcloud_vs_detections()
       Side-by-side: raw LiDAR point cloud (left) vs 3D detections (right).
       Best for explaining what PointPillars does — converts raw points
       into semantic object boxes.

    3. plot_detections_with_pointcloud()
       Combined: LiDAR point cloud as background with 3D boxes overlaid.
       Most visually impressive — boxes sit on real sensor data.
       Includes range rings, heading lines, forward direction arrow.

BEV coordinate system:
    - Origin (0,0) = ego vehicle position
    - X axis = right of vehicle
    - Y axis = forward (vehicle driving direction)
    - All distances in meters
    - Detection range: ±50m in X and Y

Box representation:
    Each 3D box has 7 parameters: [x, y, z, width, length, height, yaw]
    In BEV we show the top-down projection: rotated rectangle at (x,y)
    with width×length dimensions and yaw rotation.
    A heading line from box center to front edge shows object orientation.

Usage:
    from bev_visualization import (
        plot_detections_bev,
        plot_pointcloud_vs_detections,
        plot_detections_with_pointcloud
    )

    # After running PointPillars inference:
    plot_detections_with_pointcloud(
        boxes_np, scores_np, labels_np,
        lidar_path='/data/sets/nuscenes/samples/LIDAR_TOP/xxx.pcd.bin',
        save_path='./bev_detection.jpg'
    )
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


# ── Class definitions ─────────────────────────────────────────────────────────

CLASS_NAMES = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
    'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]

CLASS_COLORS = {
    'car':                  'cyan',
    'truck':                'orange',
    'bus':                  'yellow',
    'pedestrian':           'red',
    'motorcycle':           'lime',
    'bicycle':              'lime',
    'traffic_cone':         'white',
    'barrier':              'magenta',
    'construction_vehicle': 'orange',
    'trailer':              'orange',
}

SCORE_THRESH = 0.3


# ── Helper functions ──────────────────────────────────────────────────────────

def get_box_corners(x, y, w, l, yaw):
    """
    Compute 4 corner coordinates of a 2D rotated rectangle.

    Args:
        x, y : center position (meters)
        w, l : width and length (meters)
        yaw  : rotation angle (radians)

    Returns:
        np.ndarray: shape (4, 2) corner coordinates
    """
    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    corners = np.array([
        [-l/2, -w/2],
        [ l/2, -w/2],
        [ l/2,  w/2],
        [-l/2,  w/2]
    ])
    rot     = np.array([[cos_yaw, -sin_yaw],
                        [sin_yaw,  cos_yaw]])
    corners = (rot @ corners.T).T + np.array([x, y])
    return corners


def filter_detections(boxes, scores, labels, score_thresh=SCORE_THRESH):
    """Filter detections by confidence score threshold."""
    mask      = scores > score_thresh
    return boxes[mask], scores[mask], labels[mask]


def draw_ego_vehicle(ax, style='fancy'):
    """Draw ego vehicle rectangle at origin."""
    if style == 'fancy':
        ego = patches.FancyBboxPatch(
            (-1, -2), 2, 4,
            boxstyle="round,pad=0.1",
            edgecolor='white', facecolor='#333333',
            linewidth=2, zorder=7
        )
    else:
        ego = patches.Rectangle((-1, -2), 2, 4, color='white', zorder=5)

    ax.add_patch(ego)
    ax.text(0, 0, 'EGO',
            color='white' if style == 'fancy' else 'black',
            ha='center', va='center',
            fontsize=8, fontweight='bold',
            zorder=8 if style == 'fancy' else 6)


def draw_range_rings(ax, ranges=(10, 20, 30, 40, 50)):
    """Draw circular range rings at specified distances."""
    for r in ranges:
        circle = plt.Circle((0, 0), r, color='#333333',
                             fill=False, linewidth=0.5, zorder=2)
        ax.add_patch(circle)
        ax.text(r + 0.5, 0.5, f'{r}m',
                color='#555555', fontsize=7, zorder=2)


def draw_detection_boxes(ax, boxes_np, scores_np, labels_np,
                         show_heading=True, alpha=0.25):
    """
    Draw all detected 3D boxes as rotated 2D rectangles in BEV.

    Args:
        ax           : matplotlib axis
        boxes_np     : (N, 7+) array of box parameters [x,y,z,w,l,h,yaw,...]
        scores_np    : (N,) confidence scores
        labels_np    : (N,) class indices
        show_heading : draw heading line from center to front edge
        alpha        : box fill transparency
    """
    for i in range(len(boxes_np)):
        x, y, z, w, l, h, yaw = boxes_np[i][:7]
        cls_name = (CLASS_NAMES[labels_np[i]]
                    if labels_np[i] < len(CLASS_NAMES) else 'unknown')
        color    = CLASS_COLORS.get(cls_name, 'white')
        score    = scores_np[i]

        corners = get_box_corners(x, y, w, l, yaw)

        poly = patches.Polygon(
            corners, closed=True,
            edgecolor=color, facecolor=color,
            alpha=alpha, linewidth=2, zorder=4
        )
        ax.add_patch(poly)

        # Heading line — front of box (corners 1-2 are the front edge)
        if show_heading:
            front_center = np.mean(corners[1:3], axis=0)
            ax.plot([x, front_center[0]], [y, front_center[1]],
                    color=color, linewidth=1.5, alpha=0.9, zorder=5)

        # Class label + score
        ax.text(x, y, f'{cls_name[:3]}\n{score:.2f}',
                color=color, fontsize=6.5,
                ha='center', va='center',
                fontweight='bold', zorder=6)


def style_bev_axis(ax, title='', dark=True):
    """Apply consistent dark BEV styling to a matplotlib axis."""
    bg = '#0a0a0a' if dark else 'black'
    ax.set_facecolor(bg)
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_xlabel('X (meters)', color='#aaaaaa', fontsize=11)
    ax.set_ylabel('Y (meters)', color='#aaaaaa', fontsize=11)
    if title:
        ax.set_title(title, color='white', fontsize=13, pad=15)
    ax.tick_params(colors='#888888')
    for spine in ax.spines.values():
        spine.set_edgecolor('#333333')


# ── Visualization functions ───────────────────────────────────────────────────

def plot_detections_bev(boxes_np, scores_np, labels_np,
                        score_thresh=SCORE_THRESH,
                        save_path='/content/bev_detection.jpg'):
    """
    Basic BEV visualization — 3D detection boxes on black background.
    Fast and clean. Good for quick inspection of inference results.

    Args:
        boxes_np    : (N, 7+) box parameters from PointPillars
        scores_np   : (N,) confidence scores
        labels_np   : (N,) class label indices
        score_thresh: minimum confidence to display
        save_path   : output image path
    """
    b, s, l = filter_detections(boxes_np, scores_np, labels_np, score_thresh)
    print(f"Detections above {score_thresh} threshold: {len(b)}")

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    fig.patch.set_facecolor('black')
    style_bev_axis(ax, "PointPillars 3D Detection — Bird's Eye View")
    ax.grid(True, color='gray', alpha=0.3)

    draw_detection_boxes(ax, b, s, l, show_heading=False, alpha=0.3)
    draw_ego_vehicle(ax, style='simple')

    # Legend — only show detected classes
    legend_items = [patches.Patch(color=c, label=n)
                    for n, c in CLASS_COLORS.items()
                    if any(CLASS_NAMES[li] == n for li in l)]
    ax.legend(handles=legend_items, loc='upper right',
              facecolor='black', labelcolor='white', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='black')
    plt.show()

    print(f"✅ BEV saved → {save_path}")
    print(f"\nDetection summary:")
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        count = (l == cls_id).sum()
        if count > 0:
            print(f"  {cls_name:<25} {count}")


def plot_pointcloud_vs_detections(boxes_np, scores_np, labels_np,
                                  lidar_path,
                                  score_thresh=SCORE_THRESH,
                                  save_path='/content/pointcloud_vs_detections.jpg'):
    """
    Side-by-side visualization:
        Left  — Raw LiDAR point cloud colored by height
        Right — PointPillars 3D detections

    Best for explaining what PointPillars does:
    "Left is what the LiDAR sees. Right is what the network understands."

    Args:
        boxes_np   : (N, 7+) box parameters
        scores_np  : (N,) confidence scores
        labels_np  : (N,) class label indices
        lidar_path : path to .pcd.bin LiDAR file for point cloud background
        save_path  : output image path
    """
    b, s, l = filter_detections(boxes_np, scores_np, labels_np, score_thresh)

    # Load raw point cloud
    points      = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
    px, py, pz  = points[:, 0], points[:, 1], points[:, 2]

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.patch.set_facecolor('black')

    # Left: raw point cloud BEV colored by height
    ax1 = axes[0]
    ax1.set_facecolor('black')
    ax1.scatter(px, py, s=0.3, c=pz, cmap='viridis', alpha=0.6)
    ax1.set_xlim(-50, 50)
    ax1.set_ylim(-50, 50)
    ax1.set_title('Raw LiDAR Point Cloud (BEV)', color='white')
    ax1.set_xlabel('X (meters)', color='white')
    ax1.set_ylabel('Y (meters)', color='white')
    ax1.tick_params(colors='white')
    ax1.plot(0, 0, 'ws', markersize=8)
    ax1.text(0, -3, 'EGO', color='white', ha='center', fontsize=8)

    # Right: PointPillars detections
    ax2 = axes[1]
    ax2.set_facecolor('black')
    ax2.set_xlim(-50, 50)
    ax2.set_ylim(-50, 50)
    ax2.set_title('PointPillars 3D Detections (BEV)', color='white')
    ax2.set_xlabel('X (meters)', color='white')
    ax2.tick_params(colors='white')
    ax2.grid(True, color='gray', alpha=0.3)

    draw_detection_boxes(ax2, b, s, l, show_heading=False, alpha=0.4)
    draw_ego_vehicle(ax2, style='simple')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150,
                bbox_inches='tight', facecolor='black')
    plt.show()
    print(f"✅ Saved → {save_path}")


def plot_detections_with_pointcloud(boxes_np, scores_np, labels_np,
                                    lidar_path,
                                    score_thresh=SCORE_THRESH,
                                    save_path='/content/bev_detection_v2.jpg'):
    """
    Combined visualization — LiDAR point cloud as background with
    3D detection boxes overlaid. Most visually impressive output.

    Features:
        - LiDAR points colored by height (plasma colormap)
        - Range rings at 10/20/30/40/50m (like radar display)
        - Forward direction arrow
        - Rotated detection boxes with heading lines
        - Only detected classes shown in legend

    Args:
        boxes_np   : (N, 7+) box parameters
        scores_np  : (N,) confidence scores
        labels_np  : (N,) class label indices
        lidar_path : path to .pcd.bin LiDAR file
        save_path  : output image path
    """
    b, s, l = filter_detections(boxes_np, scores_np, labels_np, score_thresh)

    # Load raw point cloud
    points     = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)
    px, py, pz = points[:, 0], points[:, 1], points[:, 2]

    fig, ax = plt.subplots(1, 1, figsize=(14, 14))
    fig.patch.set_facecolor('#0a0a0a')
    style_bev_axis(
        ax,
        "PointPillars 3D Detection — Bird's Eye View\n"
        "nuScenes Mini | Pre-trained weights (mAP 0.354 on full val)"
    )

    # Background: LiDAR point cloud colored by height
    mask_z = (pz > -3) & (pz < 2)
    ax.scatter(px[mask_z], py[mask_z], s=0.2,
               c=pz[mask_z], cmap='plasma',
               alpha=0.4, vmin=-2, vmax=1, zorder=1)

    # Range rings
    draw_range_rings(ax)

    # Forward direction arrow
    ax.annotate('', xy=(0, 8), xytext=(0, 3),
                arrowprops=dict(arrowstyle='->', color='#888888', lw=1.5),
                zorder=3)

    # Detection boxes with heading lines
    draw_detection_boxes(ax, b, s, l, show_heading=True, alpha=0.25)

    # Ego vehicle
    draw_ego_vehicle(ax, style='fancy')

    # Legend — only detected classes
    legend_items = [patches.Patch(color=c, label=n)
                    for n, c in CLASS_COLORS.items()
                    if any(CLASS_NAMES[li] == n for li in l)]
    ax.legend(handles=legend_items, loc='upper left',
              facecolor='#1a1a1a', labelcolor='white',
              fontsize=9, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150,
                bbox_inches='tight', facecolor='#0a0a0a')
    plt.show()
    print(f"✅ Saved → {save_path}")


# ── Main — demo all three visualizations ─────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='BEV visualization for PointPillars detections'
    )
    parser.add_argument('--lidar_path',  required=True,
                        help='Path to .pcd.bin LiDAR file')
    parser.add_argument('--checkpoint',
                        default='./pointpillars_weights/pointpillars_nuscenes.pth')
    parser.add_argument('--config',
                        default='./mmdetection3d/configs/pointpillars/'
                                'pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py')
    parser.add_argument('--output_dir', default='./bev_outputs')
    parser.add_argument('--score_thresh', type=float, default=0.3)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load model and run inference
    import sys
    sys.path.insert(0, './mmdetection3d')
    from mmdet3d.apis import init_model, inference_detector
    from mmdet3d.utils import register_all_modules
    register_all_modules()

    model      = init_model(args.config, args.checkpoint, device=args.device)
    result, _  = inference_detector(model, args.lidar_path)

    boxes  = result.pred_instances_3d.bboxes_3d.tensor.cpu().numpy()
    scores = result.pred_instances_3d.scores_3d.cpu().numpy()
    labels = result.pred_instances_3d.labels_3d.cpu().numpy()

    # Generate all three visualizations
    plot_detections_bev(
        boxes, scores, labels,
        score_thresh=args.score_thresh,
        save_path=f'{args.output_dir}/bev_detections.jpg'
    )

    plot_pointcloud_vs_detections(
        boxes, scores, labels,
        lidar_path=args.lidar_path,
        score_thresh=args.score_thresh,
        save_path=f'{args.output_dir}/pointcloud_vs_detections.jpg'
    )

    plot_detections_with_pointcloud(
        boxes, scores, labels,
        lidar_path=args.lidar_path,
        score_thresh=args.score_thresh,
        save_path=f'{args.output_dir}/bev_with_pointcloud.jpg'
    )
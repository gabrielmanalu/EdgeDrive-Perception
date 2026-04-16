"""
YOLO26n-seg Segmentation Training on nuScenes
==============================================
Fine-tunes YOLO26n-seg (pretrained on COCO) on nuScenes front-camera data
for simultaneous object detection + instance segmentation.

What segmentation adds over detection:
    Detection  → 2D bounding boxes (class + position + size)
    Segmentation → 2D bounding boxes + pixel-level instance masks

    In autonomous driving context:
        - More precise object boundaries for path planning
        - Distinguishes overlapping objects (two pedestrians close together)
        - Feeds into downstream panoptic segmentation for drivable area

Additional loss terms vs detection:
    box_loss : bounding box localization (same as detection)
    cls_loss : class prediction (same as detection)
    dfl_loss : distribution focal loss (same as detection)
    seg_loss : mask prediction — how well polygon masks fit objects
    sem_loss : semantic consistency loss across instances

Label quality note:
    Training labels are convex hull polygons projected from 3D bounding boxes
    (see convert_nuscenes_seg.py), NOT pixel-perfect masks. nuScenes does not
    provide 2D segmentation annotations. Mask quality is therefore approximate —
    objects show filled rectangular/trapezoidal regions rather than true
    pixel-level outlines. For production masks, nuScenes-panoptic is required.

Segmentation strategy in this project:
    Fine-tuned YOLO26n-seg : detects nuScenes classes (car, barrier, etc.)
                             approximate masks from 3D box projection
    Base COCO YOLO26n-seg  : pixel-perfect masks on common classes
                             (car, pedestrian, bicycle) — used for visual demo
    Production goal        : fine-tune on nuScenes-panoptic for both

Results (nuScenes Mini, 100 epochs):
    Box  mAP50     : 0.594
    Box  mAP50-95  : 0.360
    Mask mAP50     : 0.484
    Mask mAP50-95  : 0.218
    Training       : ~70 min on Tesla T4

    Note: Box mAP50 (0.594) > YOLO26n-det (0.558) — the richer polygon
    labels provided additional spatial supervision during training.

Usage:
    python train_yolo26n_seg.py \\
        --data ./data/nuscenes_seg/nuscenes_seg.yaml \\
        --project ./runs \\
        --epochs 100 \\
        --batch 16 \\
        --device 0
"""

import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fine-tune YOLO26n-seg on nuScenes segmentation dataset'
    )
    parser.add_argument('--data',    default='./data/nuscenes_seg/nuscenes_seg.yaml',
                        help='Path to seg dataset YAML (from convert_nuscenes_seg.py)')
    parser.add_argument('--project', default='./runs',
                        help='Directory to save training runs')
    parser.add_argument('--name',    default='yolo26n_seg_nuscenes',
                        help='Run name (subfolder inside --project)')
    parser.add_argument('--epochs',  type=int, default=100)
    parser.add_argument('--batch',   type=int, default=16,
                        help='Batch size (reduce to 8 if OOM — seg uses more memory than det)')
    parser.add_argument('--imgsz',   type=int, default=640)
    parser.add_argument('--device',  default='0',
                        help='GPU device index, or "cpu"')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--workers', type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()

    # Load COCO-pretrained seg weights
    # -seg variant includes mask prediction head on top of detection head
    model = YOLO("yolo26n-seg.pt")

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        project=args.project,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        save=True,
        plots=True,
    )

    print(f"\n✅ Training complete")
    print(f"   Best weights  : {args.project}/{args.name}/weights/best.pt")
    print(f"   Box  mAP50    : {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.3f}")
    print(f"   Box  mAP50-95 : {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.3f}")
    print(f"   Mask mAP50    : {results.results_dict.get('metrics/mAP50(M)', 'N/A'):.3f}")
    print(f"   Mask mAP50-95 : {results.results_dict.get('metrics/mAP50-95(M)', 'N/A'):.3f}")


if __name__ == '__main__':
    main()
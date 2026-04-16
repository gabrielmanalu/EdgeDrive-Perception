"""
YOLO26n Detection Training on nuScenes
=======================================
Fine-tunes YOLO26n (pretrained on COCO) on nuScenes front-camera data
for autonomous driving object detection.

Fine-tuning vs training from scratch:
    YOLO("yolo26n.pt") loads COCO-pretrained weights — the model already
    knows edges, textures, shapes, and what cars/people look like generically.
    Fine-tuning adapts these weights to nuScenes-specific:
        - Forward-facing dashcam angle
        - Driving scene context (road, traffic lights, intersections)
        - Our 8 target classes instead of COCO's 80

    The model head is rebuilt for nc=8 (from nc=80) and ~102 weight tensors
    are re-initialized. The remaining 606/708 tensors are transferred from
    COCO weights. This is why mAP starts at ~0.02 after epoch 1 rather
    than near-zero — the backbone already understands visual features.

YOLO26 architectural advantages over YOLOv8 (visible at deployment):
    - NMS-free detection head (End2End) → lower latency variance on Jetson
    - No DFL (Distribution Focal Loss) → cleaner TensorRT export graph
    - Better INT8 quantization robustness → less accuracy drop at deployment
    - Simpler C++ decoder → no NMS post-processing needed

Note on mAP vs YOLOv8:
    YOLOv8n achieves higher mAP50 on nuScenes Mini (0.671 vs 0.558).
    This is expected — YOLO26's NMS-free head is harder to train on small
    datasets (323 images). Its advantages appear at TensorRT deployment
    on Jetson, not in small-dataset training benchmarks.

Results (nuScenes Mini, 100 epochs):
    mAP50     : 0.558
    mAP50-95  : 0.343
    Precision : 0.652
    Recall    : 0.483
    Training  : ~70 min on Tesla T4

Usage:
    python train_yolo26n.py \\
        --data ./data/nuscenes_det/nuscenes.yaml \\
        --project ./runs \\
        --epochs 100 \\
        --batch 16 \\
        --device 0
"""

import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fine-tune YOLO26n on nuScenes detection dataset'
    )
    parser.add_argument('--data',    default='./data/nuscenes_det/nuscenes.yaml',
                        help='Path to dataset YAML (from convert_nuscenes_det.py)')
    parser.add_argument('--project', default='./runs',
                        help='Directory to save training runs')
    parser.add_argument('--name',    default='yolo26n_nuscenes',
                        help='Run name (subfolder inside --project)')
    parser.add_argument('--epochs',  type=int, default=100)
    parser.add_argument('--batch',   type=int, default=16,
                        help='Batch size (reduce to 8 if OOM on smaller GPUs)')
    parser.add_argument('--imgsz',   type=int, default=640)
    parser.add_argument('--device',  default='0',
                        help='GPU device index, or "cpu"')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (epochs without improvement)')
    parser.add_argument('--workers', type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()

    # Load COCO-pretrained weights
    # Using .pt file (not .yaml) is what triggers fine-tuning vs from-scratch
    model = YOLO("yolo26n.pt")

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
        plots=True,   # saves training curve plots to run directory
    )

    print(f"\n✅ Training complete")
    print(f"   Best weights : {args.project}/{args.name}/weights/best.pt")
    print(f"   mAP50        : {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.3f}")
    print(f"   mAP50-95     : {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.3f}")


if __name__ == '__main__':
    main()
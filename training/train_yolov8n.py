"""
YOLOv8n Detection Training on nuScenes (Baseline)
==================================================
Fine-tunes YOLOv8n on nuScenes front-camera data as a baseline for
direct comparison against YOLO26n under identical training conditions.

Purpose of this model:
    This is NOT the primary deployment model — it exists purely for
    generational comparison. Identical settings (epochs, batch, imgsz,
    dataset) isolate architectural differences between YOLOv8 and YOLO26.

Why YOLOv8n outperforms YOLO26n on nuScenes Mini:
    YOLOv8n achieved mAP50 0.671 vs YOLO26n's 0.558 on this dataset.
    This is expected and explained by:

    1. More mature COCO pretraining — YOLOv8 has been refined since 2023.
       Better starting weights → better fine-tuning on small datasets.

    2. Traditional NMS head converges faster on small data — YOLO26's
       NMS-free head needs more examples to learn confident predictions
       without the NMS filtering step as a safety net.

    3. cls_loss comparison:
       YOLOv8n  cls_loss @ epoch 100: 0.709  (much lower)
       YOLO26n  cls_loss @ epoch 100: 1.191  (higher)
       Confirms YOLO26's head is harder to train on 323 images.

Where YOLO26n wins (deployment on Jetson Orin Nano):
    Metric                    YOLO26n     YOLOv8n
    TensorRT inference          faster      slower
    INT8 accuracy drop          smaller     larger
    Latency variance P99-P50    tighter     wider
    C++ decoder complexity      simpler     needs NMS
    Power consumption           lower       higher

The mAP gap closes significantly with more data. On the full nuScenes
dataset (28k samples), YOLO26n is expected to match or exceed YOLOv8n.

Results (nuScenes Mini, 100 epochs):
    mAP50     : 0.671
    mAP50-95  : 0.409
    Precision : 0.763
    Recall    : 0.609
    Training  : ~70 min on Tesla T4

Usage:
    python train_yolov8n.py \\
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
        description='Fine-tune YOLOv8n on nuScenes as YOLO26n baseline comparison'
    )
    parser.add_argument('--data',    default='./data/nuscenes_det/nuscenes.yaml',
                        help='Path to dataset YAML — same as YOLO26n for fair comparison')
    parser.add_argument('--project', default='./runs',
                        help='Directory to save training runs')
    parser.add_argument('--name',    default='yolov8n_nuscenes',
                        help='Run name (subfolder inside --project)')
    parser.add_argument('--epochs',  type=int, default=100,
                        help='Must match YOLO26n training for fair comparison')
    parser.add_argument('--batch',   type=int, default=16,
                        help='Must match YOLO26n training for fair comparison')
    parser.add_argument('--imgsz',   type=int, default=640,
                        help='Must match YOLO26n training for fair comparison')
    parser.add_argument('--device',  default='0',
                        help='GPU device index, or "cpu"')
    parser.add_argument('--patience', type=int, default=20,
                        help='Must match YOLO26n training for fair comparison')
    parser.add_argument('--workers', type=int, default=2)
    return parser.parse_args()


def main():
    args = parse_args()

    # Load COCO-pretrained YOLOv8n weights
    model = YOLO("yolov8n.pt")

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
    print(f"   Best weights : {args.project}/{args.name}/weights/best.pt")
    print(f"   mAP50        : {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.3f}")
    print(f"   mAP50-95     : {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.3f}")
    print(f"\n   See docs/yolo26_vs_yolov8.md for full comparison analysis")


if __name__ == '__main__':
    main()
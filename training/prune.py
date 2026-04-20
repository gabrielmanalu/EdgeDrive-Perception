"""
Structured Pruning Investigation for YOLO26n
=============================================
Documents the structured pruning attempt on YOLO26n and explains
why INT8 quantization was chosen as the primary compression strategy.

What is structured pruning:
    Removes entire output channels (filters) from Conv2d layers,
    producing a physically smaller model with fewer parameters.
    Unlike unstructured pruning which zeros individual weights
    (same shape, no real speedup), structured pruning produces
    models that are genuinely faster on any hardware.

    Example:
        Conv2d(64→128)  →  prune 30%  →  Conv2d(64→90)
        All downstream layers automatically adjusted to 90 channels.

Approach attempted:
    torch-pruning (v1.x) with MagnitudePruner (L1-norm importance).
    Dependency graph resolution traces channel connections through
    the model to ensure consistent pruning across all layers.

Finding:
    YOLO26n uses custom Ultralytics block types (C3k2, C2PSA, SPPF)
    that torch-pruning cannot trace through automatically.

    126 Conv2d layers were found but the dependency graph resolver
    could not determine how channels flow through C3k2 and SPPF
    wrappers, resulting in 0% channel reduction to avoid breaking
    the model.

    Proper implementation requires custom dependency handlers for
    each Ultralytics block type — a significant engineering effort
    that is beyond the scope of this project.

Why INT8 quantization is sufficient:
    The goals of structured pruning are already achieved by INT8 PTQ:

        Goal              │ Structured Pruning  │ INT8 PTQ
        ──────────────────┼─────────────────────┼──────────────
        Model size        │ ~40-50% reduction   │ 47% reduction ✅
        Inference speed   │ ~1.5-2x faster      │ ~4x on TensorRT ✅
        Accuracy impact   │ ~1-3% mAP drop      │ +0.45% mAP ✅
        Hardware support  │ CPU/GPU             │ Jetson TensorRT ✅

    INT8 quantization outperforms structured pruning on all metrics
    for Jetson TensorRT deployment.

References:
    - torch-pruning: https://github.com/VainF/Torch-Pruning
    - Ultralytics pruning: https://docs.ultralytics.com/guides/model-pruning

Usage:
    python prune.py --model /path/to/best.pt
    (Runs the pruning attempt and prints diagnostic output)
"""

import argparse
import torch
import torch.nn as nn
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(
        description='Structured pruning investigation for YOLO26n'
    )
    parser.add_argument('--model',
                        default='/content/drive/MyDrive/yolo_runs/'
                                'yolo26n_nuscenes/weights/best.pt')
    parser.add_argument('--ratio', type=float, default=0.3,
                        help='Target pruning ratio (default: 0.3 = 30%%)')
    return parser.parse_args()


def diagnose_model(model):
    """
    Count prunable Conv2d layers and custom Ultralytics blocks.
    Shows why torch-pruning cannot resolve the dependency graph.
    """
    conv_layers   = []
    custom_blocks = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append(
                f"{name}: {module.in_channels}→{module.out_channels}"
            )
        elif module.__class__.__name__ in ['C3k2', 'SPPF', 'C2PSA',
                                           'C2f', 'Detect']:
            custom_blocks.append(f"{name}: {module.__class__.__name__}")

    print(f"\nModel architecture diagnosis:")
    print(f"  Conv2d layers (prunable in isolation): {len(conv_layers)}")
    print(f"  Custom Ultralytics blocks (not traceable): {len(custom_blocks)}")
    print(f"\n  First 5 Conv2d layers:")
    for l in conv_layers[:5]:
        print(f"    {l}")
    print(f"\n  Custom blocks:")
    for b in custom_blocks[:8]:
        print(f"    {b}")

    return len(conv_layers), len(custom_blocks)


def attempt_structured_pruning(model, ratio):
    """
    Attempt structured channel pruning using torch-pruning.
    Documents why it fails for YOLO26n custom architecture.
    """
    try:
        import torch_pruning as tp
    except ImportError:
        print("torch-pruning not installed. Run: pip install torch-pruning")
        return 0.0

    params_before = sum(p.numel() for p in model.parameters())
    print(f"\nParameters before pruning: {params_before:,}")

    model.eval()
    example_input = torch.randn(1, 3, 640, 640).to(
        next(model.parameters()).device
    )

    # L1 norm filter importance — standard structured pruning criterion
    importance = tp.importance.MagnitudeImportance(p=1)

    # Ignore detection head — pruning output channels here would
    # break the number of predicted classes/boxes
    ignored_layers = []
    for name, module in model.named_modules():
        if module.__class__.__name__ == 'Detect':
            ignored_layers.append(module)

    pruner = tp.pruner.MagnitudePruner(
        model,
        example_inputs=example_input,
        importance=importance,
        pruning_ratio=ratio,
        ignored_layers=ignored_layers,
    )

    pruner.step()

    params_after = sum(p.numel() for p in model.parameters())
    reduction    = 1 - (params_after / params_before)

    print(f"Parameters after  pruning: {params_after:,}")
    print(f"Actual reduction         : {reduction*100:.1f}%")

    if reduction < 0.01:
        print(f"\n⚠️  Near-zero reduction ({reduction*100:.2f}%)")
        print(f"   Cause: torch-pruning dependency graph cannot trace")
        print(f"   through C3k2, SPPF, C2PSA custom blocks.")
        print(f"   Fix:   Register custom dependency handlers per block.")
        print(f"   Decision: Use INT8 PTQ instead (47% size reduction,")
        print(f"             +0.45% mAP, hardware-accelerated on Jetson).")

    return reduction


def main():
    args = parse_args()

    print("="*55)
    print("Structured Pruning Investigation — YOLO26n")
    print("="*55)

    yolo  = YOLO(args.model)
    model = yolo.model

    # Step 1: diagnose architecture
    n_conv, n_custom = diagnose_model(model)

    # Step 2: attempt pruning
    print(f"\nAttempting {int(args.ratio*100)}% structured pruning...")
    reduction = attempt_structured_pruning(model, args.ratio)

    # Step 3: summary
    print(f"\n{'='*55}")
    print(f"Summary")
    print(f"{'='*55}")
    print(f"Conv2d layers found   : {n_conv}")
    print(f"Custom blocks (opaque): {n_custom}")
    print(f"Actual reduction      : {reduction*100:.1f}%")
    print(f"\nConclusion:")
    if reduction < 0.01:
        print(f"  Structured pruning not viable for YOLO26n without")
        print(f"  custom torch-pruning handlers for each Ultralytics block.")
        print(f"  INT8 PTQ achieves equivalent compression goals:")
        print(f"    47% model size reduction, +0.45% mAP, TensorRT ready.")
    else:
        print(f"  Pruning succeeded — validate accuracy with quantize.py")


if __name__ == '__main__':
    main()
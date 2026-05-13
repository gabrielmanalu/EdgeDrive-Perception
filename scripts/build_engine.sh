#!/bin/bash
# build_engine.sh — Export TensorRT engines from PyTorch weights
#
# Usage:
#   ./scripts/build_engine.sh [fp16|int8|all]
#
# Examples:
#   ./scripts/build_engine.sh fp16       # FP16 only
#   ./scripts/build_engine.sh int8       # INT8 only
#   ./scripts/build_engine.sh all        # FP16 + INT8 (default)
#
# Requirements:
#   - Run on Jetson (TensorRT engines are hardware-specific)
#   - weights/yolo26n_det.pt must exist
#   - calibration.yaml + calibration_images/ for INT8
#   - ~10 min for FP16, ~7 min for INT8
#
# Output:
#   weights/yolo26n_det_fp16.engine      (Ultralytics wrapper)
#   weights/yolo26n_det_int8.engine      (Ultralytics wrapper)
#   weights/yolo26n_det_fp16_raw.engine  (raw TRT, for C++ pipeline)
#   weights/yolo26n_det_int8_raw.engine  (raw TRT, for C++ pipeline)

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-all}"
WEIGHTS="weights/yolo26n_det.pt"
CALIB_YAML="calibration.yaml"

# ── Checks ────────────────────────────────────────────────────────────────────

if [ ! -f "$WEIGHTS" ]; then
    echo "Error: $WEIGHTS not found."
    echo "Download from Google Drive or run training first."
    exit 1
fi

if [[ "$MODE" == "int8" || "$MODE" == "all" ]]; then
    if [ ! -f "$CALIB_YAML" ]; then
        echo "Error: $CALIB_YAML not found (required for INT8)."
        exit 1
    fi
    if [ ! -d "calibration_images" ]; then
        echo "Error: calibration_images/ not found."
        echo "Run the calibration image extraction script first:"
        echo "  python3 -c \"\$(grep -A20 'To reproduce' calibration.yaml | grep -v '#')\""
        exit 1
    fi
fi

echo "================================================="
echo "EdgeDrive Perception — TensorRT Engine Builder"
echo "================================================="
echo "Mode   : $MODE"
echo "Weights: $WEIGHTS"
echo ""

# ── Helper: extract raw engine from Ultralytics wrapper ───────────────────────

extract_raw() {
    local wrapper="$1"
    local raw="$2"
    echo "Extracting raw engine: $raw"
    python3 - << PYEOF
import struct
with open('$wrapper', 'rb') as f:
    content = f.read()
meta_len = struct.unpack('<I', content[:4])[0]
raw_engine = content[4 + meta_len:]
with open('$raw', 'wb') as f:
    f.write(raw_engine)
print(f'  Raw engine size: {len(raw_engine)/1024/1024:.1f} MB')
print(f'  Magic bytes: {raw_engine[:4]}')
PYEOF
}

# ── FP16 export ───────────────────────────────────────────────────────────────

build_fp16() {
    echo "-------------------------------------------------"
    echo "Building TensorRT FP16 engine..."
    echo "Estimated time: ~9 minutes"
    echo "-------------------------------------------------"

    python3 - << 'PYEOF'
from ultralytics import YOLO
model = YOLO('weights/yolo26n_det.pt')
model.export(
    format='engine',
    device=0,
    half=True,
    imgsz=640,
    workspace=4,
)
print('FP16 engine exported: weights/yolo26n_det.engine')
PYEOF

    mv weights/yolo26n_det.engine weights/yolo26n_det_fp16.engine
    echo "Saved: weights/yolo26n_det_fp16.engine"

    extract_raw weights/yolo26n_det_fp16.engine weights/yolo26n_det_fp16_raw.engine
    echo "Saved: weights/yolo26n_det_fp16_raw.engine"
}

# ── INT8 export ───────────────────────────────────────────────────────────────

build_int8() {
    echo "-------------------------------------------------"
    echo "Building TensorRT INT8 engine..."
    echo "Calibration: $CALIB_YAML ($(ls calibration_images/ | wc -l) images)"
    echo "Estimated time: ~7 minutes"
    echo "Note: end2end disabled on JetPack 6 (expected warning)"
    echo "-------------------------------------------------"

    python3 - << 'PYEOF'
import torch
torch.cuda.empty_cache()
from ultralytics import YOLO
model = YOLO('weights/yolo26n_det.pt')
model.export(
    format='engine',
    device=0,
    int8=True,
    data='calibration.yaml',
    imgsz=640,
    workspace=2,
    batch=1,
)
print('INT8 engine exported: weights/yolo26n_det.engine')
PYEOF

    mv weights/yolo26n_det.engine weights/yolo26n_det_int8.engine
    echo "Saved: weights/yolo26n_det_int8.engine"

    extract_raw weights/yolo26n_det_int8.engine weights/yolo26n_det_int8_raw.engine
    echo "Saved: weights/yolo26n_det_int8_raw.engine"
}

# ── Main ──────────────────────────────────────────────────────────────────────

case "$MODE" in
    fp16) build_fp16 ;;
    int8) build_int8 ;;
    all)  build_fp16; build_int8 ;;
    *)
        echo "Usage: $0 [fp16|int8|all]"
        exit 1
        ;;
esac

echo ""
echo "================================================="
echo "Done. Engine files:"
ls -lh weights/*.engine 2>/dev/null || echo "No .engine files found"
echo "================================================="
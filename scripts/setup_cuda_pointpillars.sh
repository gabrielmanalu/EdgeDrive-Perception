#!/bin/bash
# setup_cuda_pointpillars.sh — Clone, patch, build CUDA-PointPillars on Jetson
#
# Sets up NVIDIA-AI-IOT/CUDA-PointPillars with our nuScenes patches:
#   - TRT 10.x API (getNbIOTensors, setTensorAddress, enqueueV3)
#   - CHW scatter kernel for our backbone-only ONNX
#   - nuScenes voxelization + postprocess params (10 classes, ±50m range)
#   - C++17 fix for CUDA 12.6 compatibility
#
# Usage:
#   ./scripts/setup_cuda_pointpillars.sh
#
# Prerequisites:
#   - ONNX exported from mmdetection3d in weights/pointpillars_nuscenes_backbone.onnx
#   - JetPack R36.4.7 (CUDA 12.6, TRT 10.3.0, SM87)
#
# Output:
#   cuda-pointpillars/build/pointpillar  ← inference binary
#   cuda-pointpillars/model/pointpillar.plan  ← TRT FP16 engine
#
# Run inference after setup:
#   cd cuda-pointpillars/build
#   ./pointpillar ../data/ ../data/ --timer

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

CUDA_PP_DIR="$REPO_ROOT/cuda-pointpillars"
PATCHES_DIR="$REPO_ROOT/cuda-pointpillars-patches"
ONNX="$REPO_ROOT/weights/pointpillars_nuscenes_backbone.onnx"
TRT_BIN="/usr/src/tensorrt/bin/trtexec"

# ── Checks ────────────────────────────────────────────────────────────────────

echo ""
echo "================================================="
echo "EdgeDrive — CUDA-PointPillars Setup"
echo "================================================="

if [ ! -f "$ONNX" ]; then
    echo "Error: ONNX not found at $ONNX"
    echo "Export it from mmdetection3d first (see fusion/train_pointpillars.py)"
    exit 1
fi

if [ ! -f "$TRT_BIN" ]; then
    echo "Error: trtexec not found at $TRT_BIN"
    exit 1
fi

if [ ! -d "$PATCHES_DIR" ]; then
    echo "Error: patches not found at $PATCHES_DIR"
    exit 1
fi

echo "  ONNX    : $ONNX"
echo "  Patches : $PATCHES_DIR"
echo ""

# ── Step 1: Clone ─────────────────────────────────────────────────────────────

if [ -d "$CUDA_PP_DIR" ]; then
    echo "[1/5] cuda-pointpillars/ already exists — skipping clone"
else
    echo "[1/5] Cloning NVIDIA-AI-IOT/CUDA-PointPillars..."
    git clone https://github.com/NVIDIA-AI-IOT/CUDA-PointPillars.git \
        "$CUDA_PP_DIR"
    echo "      ✅ cloned"
fi

# ── Step 2: Apply patches ─────────────────────────────────────────────────────

echo "[2/5] Applying nuScenes patches..."

# C++17 fix
sed -i 's/-std=c++14/-std=c++17/g' "$CUDA_PP_DIR/CMakeLists.txt"

# Apply our 4 patched files
cp "$PATCHES_DIR/tensorrt.cpp"         "$CUDA_PP_DIR/src/common/tensorrt.cpp"
cp "$PATCHES_DIR/lidar-backbone.cu"    "$CUDA_PP_DIR/src/pointpillar/lidar-backbone.cu"
cp "$PATCHES_DIR/main.cpp"             "$CUDA_PP_DIR/src/main.cpp"
cp "$PATCHES_DIR/lidar-postprocess.hpp" "$CUDA_PP_DIR/src/pointpillar/lidar-postprocess.hpp"

echo "      ✅ patches applied"

# ── Step 3: Build ─────────────────────────────────────────────────────────────

echo "[3/5] Building CUDA-PointPillars (SM87)..."

export CUDASM=87
export TensorRT_Lib=/usr/lib/aarch64-linux-gnu/
export TensorRT_Inc=/usr/include/aarch64-linux-gnu/
export TensorRT_Bin=/usr/src/tensorrt/bin/
export CUDA_Lib=/usr/local/cuda-12.6/targets/aarch64-linux/lib/
export CUDA_Inc=/usr/local/cuda-12.6/targets/aarch64-linux/include/
export CUDA_Bin=/usr/local/cuda-12.6/bin/
export CUDA_HOME=/usr/local/cuda-12.6/
export CUDNN_Lib=/usr/lib/aarch64-linux-gnu/
export PATH=$TensorRT_Bin:$CUDA_Bin:$PATH

mkdir -p "$CUDA_PP_DIR/build"
cd "$CUDA_PP_DIR/build"
cmake .. -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
make -j4
cd "$REPO_ROOT"

echo "      ✅ binary built: cuda-pointpillars/build/pointpillar"

# ── Step 4: Build TRT engine ──────────────────────────────────────────────────

echo "[4/5] Building TRT FP16 engine from ONNX..."

cp "$ONNX" "$CUDA_PP_DIR/model/pointpillar_nuscenes_backbone.onnx"

$TRT_BIN \
    --onnx="$CUDA_PP_DIR/model/pointpillar_nuscenes_backbone.onnx" \
    --fp16 \
    --plugins="$CUDA_PP_DIR/build/libpointpillar_core.so" \
    --saveEngine="$CUDA_PP_DIR/model/pointpillar.plan" \
    2>&1 | grep -E "PASSED|FAILED|latency|Throughput"

if [ ! -f "$CUDA_PP_DIR/model/pointpillar.plan" ]; then
    echo "Error: TRT engine build failed"
    exit 1
fi

echo "      ✅ engine: cuda-pointpillars/model/pointpillar.plan"

# ── Step 5: Test ──────────────────────────────────────────────────────────────

echo "[5/5] Testing with sample KITTI data..."

cd "$CUDA_PP_DIR/build"
./pointpillar ../data/ ../data/ --timer 2>&1 | \
    grep -E "Total|Detections|Backbone|Voxel"

cd "$REPO_ROOT"

echo ""
echo "================================================="
echo "✅ CUDA-PointPillars ready"
echo ""
echo "Run on nuScenes LiDAR data:"
echo "  cp /data/sets/nuscenes/samples/LIDAR_TOP/<frame>.pcd.bin \\"
echo "     cuda-pointpillars/data/"
echo "  cd cuda-pointpillars/build"
echo "  ./pointpillar ../data/ ../data/ --timer"
echo "================================================="
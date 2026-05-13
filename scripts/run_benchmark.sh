#!/bin/bash
# run_benchmark.sh — Run full benchmark suite on Jetson
#
# Usage:
#   ./scripts/run_benchmark.sh [python|cpp|all]
#
# Examples:
#   ./scripts/run_benchmark.sh python    # Python FP32/FP16/INT8
#   ./scripts/run_benchmark.sh cpp       # C++ TRT INT8
#   ./scripts/run_benchmark.sh all       # everything (default)
#
# Requirements:
#   - sudo jetson_clocks must be run before this script
#     (script will warn if clocks are not at max)
#   - weights/*.engine must exist (run build_engine.sh first)
#   - test_images/ must exist (404 nuScenes CAM_FRONT images)
#   - deployment/build/edgedrive must be compiled (C++ only)
#
# Output:
#   Prints results to stdout.
#   Run with: ./scripts/run_benchmark.sh 2>&1 | tee benchmarks/results/run_$(date +%Y%m%d).txt

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-all}"
DURATION=60
IMAGES_DIR="test_images"
CPP_BIN="deployment/build/edgedrive"
INT8_ENGINE="weights/yolo26n_det_int8_raw.engine"

# ── Checks ────────────────────────────────────────────────────────────────────

check_clocks() {
    # Check if GPU is at max clock (jetson_clocks sets GPU to 1017+ MHz)
    GPU_FREQ=$(cat /sys/devices/17000000.ga10b/devfreq/17000000.ga10b/cur_freq 2>/dev/null || echo "0")
    if [ "$GPU_FREQ" -lt "900000000" ] 2>/dev/null; then
        echo "WARNING: GPU clock may not be at maximum."
        echo "  Run 'sudo jetson_clocks' for consistent benchmark results."
        echo "  Current GPU freq: $(echo "scale=0; $GPU_FREQ/1000000" | bc) MHz"
        echo ""
        read -p "Continue anyway? [y/N] " -n 1 -r
        echo
        [[ $REPLY =~ ^[Yy]$ ]] || exit 1
    else
        echo "GPU clocks: OK ($(echo "scale=0; $GPU_FREQ/1000000" | bc) MHz)"
    fi
}

check_images() {
    if [ ! -d "$IMAGES_DIR" ]; then
        echo "Error: $IMAGES_DIR not found."
        echo "Copy nuScenes CAM_FRONT images to $IMAGES_DIR/"
        exit 1
    fi
    COUNT=$(ls "$IMAGES_DIR"/*.jpg 2>/dev/null | wc -l)
    if [ "$COUNT" -lt 10 ]; then
        echo "Error: Only $COUNT images found in $IMAGES_DIR/. Need at least 10."
        exit 1
    fi
    echo "Test images: $COUNT images in $IMAGES_DIR/"
}

# ── Python benchmark ──────────────────────────────────────────────────────────

run_python() {
    echo ""
    echo "================================================="
    echo "Python Benchmark (FP32 / FP16 / INT8)"
    echo "Duration: ${DURATION}s each, 404 images preloaded"
    echo "================================================="

    python3 - << PYEOF
import glob, time, numpy as np
from ultralytics import YOLO

images = sorted(glob.glob('$IMAGES_DIR/*.jpg'))
print(f'Loaded {len(images)} images')
DURATION = $DURATION

def benchmark(name, model_path, half=False, int8=False):
    print(f'\nLoading {name} model: {model_path} ...')
    model = YOLO(model_path)

    print('Warming up (10 frames)...')
    for i in range(10):
        model(images[i % len(images)], device=0, verbose=False,
              half=half)

    print(f'Running {name} benchmark for {DURATION} seconds...')
    pre_times, inf_times, post_times = [], [], []
    start = time.time()
    i = 0
    while time.time() - start < DURATION:
        r = model(images[i % len(images)], device=0, verbose=False,
                  half=half)
        pre_times.append(r[0].speed['preprocess'])
        inf_times.append(r[0].speed['inference'])
        post_times.append(r[0].speed['postprocess'])
        i += 1

    elapsed = time.time() - start
    total = np.mean(pre_times) + np.mean(inf_times) + np.mean(post_times)

    print(f'\n=== {name} Python Benchmark Results ===')
    print(f'Frames        : {i}')
    print(f'Duration      : {elapsed:.1f}s')
    print(f'FPS (wall)    : {i/elapsed:.1f}')
    print(f'Total / frame : {total:.1f}ms (excluding Python loop overhead)')
    print(f'  Preprocess  : {np.mean(pre_times):.1f}ms')
    print(f'  Inference   : {np.mean(inf_times):.1f}ms')
    print(f'  Postprocess : {np.mean(post_times):.1f}ms')
    print(f'=======================================')

benchmark('FP32',  'weights/yolo26n_det.pt',          half=False)
benchmark('FP16',  'weights/yolo26n_det_fp16.engine',  half=True)
benchmark('INT8',  'weights/yolo26n_det_int8.engine',  half=False)
PYEOF
}

# ── C++ benchmark ─────────────────────────────────────────────────────────────

run_cpp() {
    echo ""
    echo "================================================="
    echo "C++ TensorRT INT8 Benchmark"
    echo "Engine : $INT8_ENGINE"
    echo "Duration: ${DURATION}s, 404 images preloaded"
    echo "================================================="

    if [ ! -f "$CPP_BIN" ]; then
        echo "Error: $CPP_BIN not found."
        echo "Build the C++ pipeline first:"
        echo "  cd deployment && mkdir -p build && cd build && cmake .. && make -j4"
        exit 1
    fi

    if [ ! -f "$INT8_ENGINE" ]; then
        echo "Error: $INT8_ENGINE not found."
        echo "Run: ./scripts/build_engine.sh int8"
        exit 1
    fi

    "$CPP_BIN" \
        --engine "$INT8_ENGINE" \
        --benchmark \
        --images "$IMAGES_DIR" \
        --duration "$DURATION" \
        --threshold 0.3
}

# ── Main ──────────────────────────────────────────────────────────────────────

echo "================================================="
echo "EdgeDrive Perception — Benchmark Suite"
echo "================================================="
echo "Mode   : $MODE"
echo "Date   : $(date)"
echo ""

check_clocks
check_images

case "$MODE" in
    python) run_python ;;
    cpp)    run_cpp ;;
    all)    run_python; run_cpp ;;
    *)
        echo "Usage: $0 [python|cpp|all]"
        exit 1
        ;;
esac

echo ""
echo "Benchmark complete: $(date)"
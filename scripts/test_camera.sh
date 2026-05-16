#!/bin/bash
# test_camera.sh — Test all camera_capture modes
#
# Usage:
#   ./scripts/test_camera.sh [mode] [--bev]
#
# Modes:
#   usb        — USB webcam live inference
#   csi        — CSI camera (Jetson nvarguscamerasrc)
#   video      — Pre-recorded video file (loops)
#   save       — USB camera → output/camera_demo.mp4 (30s)
#   headless   — USB camera, no display, 10s
#   video-save — video file → annotated output (demo reel)
#   bev-save   — video file → BEV side-by-side output
#   all        — run all non-interactive tests
#
# Add --bev to any mode for BEV visualization:
#   ./scripts/test_camera.sh usb --bev
#   ./scripts/test_camera.sh video --bev
#   ./scripts/test_camera.sh save --bev
#   ./scripts/test_camera.sh bev-save      ← BEV always on here

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-usb}"
BEV_FLAG=""
for arg in "$@"; do
    [ "$arg" = "--bev" ] && BEV_FLAG="--bev"
done

ENGINE="weights/yolo26n_det_int8_raw.engine"
BIN="deployment/build/edgedrive"
THRESHOLD="0.3"

# ── Checks ────────────────────────────────────────────────────────────────────

check_bin() {
    if [ ! -f "$BIN" ]; then
        echo "Error: $BIN not found. Build first:"
        echo "  cd deployment/build && make -j4"
        exit 1
    fi
}

check_engine() {
    if [ ! -f "$ENGINE" ]; then
        echo "Error: $ENGINE not found."
        echo "  ./scripts/build_engine.sh int8"
        exit 1
    fi
}

check_video() {
    if ls test_videos/*.mp4 2>/dev/null | head -1; then
        :
    elif ls *.mp4 2>/dev/null | head -1; then
        :
    else
        echo ""
    fi
}

check_display() {
    if [ -z "$DISPLAY" ] && [ -z "$WAYLAND_DISPLAY" ]; then
        echo "Warning: No display detected. Switching to headless."
        return 1
    fi
    return 0
}

check_camera() {
    local id="${1:-0}"
    if [ ! -e "/dev/video${id}" ]; then
        echo "Error: /dev/video${id} not found."
        echo "  ls /dev/video*"
        exit 1
    fi
    echo "Camera /dev/video${id} found."
}

bev_label() { [ -n "$BEV_FLAG" ] && echo " + BEV" || echo ""; }

check_bin
check_engine

# ── Test modes ────────────────────────────────────────────────────────────────

test_usb() {
    echo ""
    echo "================================================="
    echo "Test: USB Camera Live Inference$(bev_label)"
    echo "Camera: /dev/video0 | Press 'q' or ESC to quit"
    echo "================================================="
    check_camera 0
    check_display || { test_headless; return; }
    "$BIN" --engine "$ENGINE" --camera 0 --threshold "$THRESHOLD" $BEV_FLAG
}

test_csi() {
    echo ""
    echo "================================================="
    echo "Test: CSI Camera (nvarguscamerasrc)$(bev_label)"
    echo "================================================="
    if ! gst-inspect-1.0 nvarguscamerasrc &>/dev/null; then
        echo "Error: nvarguscamerasrc not found."
        exit 1
    fi
    check_display || {
        mkdir -p output
        "$BIN" --engine "$ENGINE" --csi 0 --threshold "$THRESHOLD" \
               --no-display --save-video output/csi_test.mp4 $BEV_FLAG
        return
    }
    "$BIN" --engine "$ENGINE" --csi 0 --threshold "$THRESHOLD" $BEV_FLAG
}

test_video() {
    echo ""
    echo "================================================="
    echo "Test: Pre-recorded Video Inference$(bev_label)"
    echo "Video loops. Press 'q' or ESC to quit."
    echo "================================================="
    VIDEO=$(check_video)
    if [ -z "$VIDEO" ]; then
        echo "No video found."
        echo "  python3 scripts/make_test_video.py"
        exit 1
    fi
    echo "Video: $VIDEO"
    check_display || {
        mkdir -p output
        OUT="output/$(basename "$VIDEO" .mp4)_headless.mp4"
        "$BIN" --engine "$ENGINE" --video "$VIDEO" --threshold "$THRESHOLD" \
               --no-display --no-loop --save-video "$OUT" $BEV_FLAG
        echo "Saved: $OUT"
        return
    }
    "$BIN" --engine "$ENGINE" --video "$VIDEO" --threshold "$THRESHOLD" $BEV_FLAG
}

test_save() {
    echo ""
    echo "================================================="
    echo "Test: Save Annotated Video$(bev_label) — USB camera, 30s"
    echo "================================================="
    check_camera 0
    mkdir -p output
    local OUT="output/camera_demo${BEV_FLAG:+_bev}.mp4"
    echo "Recording 30s → $OUT"
    "$BIN" --engine "$ENGINE" --camera 0 --threshold "$THRESHOLD" \
           --save-video "$OUT" --no-display $BEV_FLAG &
    BG_PID=$!
    sleep 30
    kill $BG_PID 2>/dev/null
    wait $BG_PID 2>/dev/null || true
    [ -f "$OUT" ] && echo "✅ Saved: $OUT ($(du -h "$OUT" | cut -f1))"
}

test_headless() {
    echo ""
    echo "================================================="
    echo "Test: Headless Inference (10s)"
    echo "================================================="
    check_camera 0
    echo "Running for 10 seconds..."
    timeout 10 "$BIN" --engine "$ENGINE" --camera 0 \
               --threshold "$THRESHOLD" --no-display $BEV_FLAG || true
    echo "✅ Headless test complete."
}

test_video_save() {
    echo ""
    echo "================================================="
    echo "Test: Video → Annotated Output$(bev_label)"
    echo "================================================="
    VIDEO=$(check_video)
    if [ -z "$VIDEO" ]; then
        echo "No video found."
        echo "  python3 scripts/make_test_video.py"
        exit 1
    fi
    mkdir -p output
    local SUFFIX=""
    [ -n "$BEV_FLAG" ] && SUFFIX="_bev"
    OUT="output/demo_$(basename "$VIDEO" .mp4)${SUFFIX}_annotated.mp4"
    echo "Input : $VIDEO"
    echo "Output: $OUT"
    "$BIN" --engine "$ENGINE" --video "$VIDEO" --threshold "$THRESHOLD" \
           --no-display --no-loop --save-video "$OUT" $BEV_FLAG
    [ -f "$OUT" ] && echo "✅ Done: $OUT ($(du -h "$OUT" | cut -f1))"
}

test_bev_save() {
    echo ""
    echo "================================================="
    echo "Test: Video → BEV Side-by-Side Output"
    echo "================================================="
    VIDEO=$(check_video)
    if [ -z "$VIDEO" ]; then
        echo "No video found."
        echo "  python3 scripts/make_test_video.py"
        exit 1
    fi
    mkdir -p output
    OUT="output/demo_$(basename "$VIDEO" .mp4)_bev.mp4"
    echo "Input : $VIDEO"
    echo "Output: $OUT"
    echo "BEV side-by-side enabled."
    "$BIN" --engine "$ENGINE" --video "$VIDEO" --threshold "$THRESHOLD" \
           --no-display --no-loop --save-video "$OUT" --bev
    if [ -f "$OUT" ]; then
        SIZE=$(du -h "$OUT" | cut -f1)
        echo ""
        echo "✅ Done: $OUT ($SIZE)"
        echo "   Upload to GitHub or YouTube as BEV demo reel."
    fi
}

test_all() {
    echo "Running all non-interactive tests..."
    test_headless
    test_video_save
    BEV_FLAG="--bev" test_video_save
    test_bev_save
    test_save
    echo ""
    echo "All done. Run interactive tests manually:"
    echo "  ./scripts/test_camera.sh usb"
    echo "  ./scripts/test_camera.sh video"
    echo "  ./scripts/test_camera.sh usb --bev"
    echo "  ./scripts/test_camera.sh video --bev"
}

# ── Main ──────────────────────────────────────────────────────────────────────

case "$MODE" in
    usb)        test_usb ;;
    csi)        test_csi ;;
    video)      test_video ;;
    save)       test_save ;;
    headless)   test_headless ;;
    video-save) test_video_save ;;
    bev-save)   test_bev_save ;;
    all)        test_all ;;
    --bev)
        # Handle: ./scripts/test_camera.sh --bev (no mode specified)
        MODE="usb"
        BEV_FLAG="--bev"
        test_usb ;;
    *)
        echo "Usage: $0 [mode] [--bev]"
        echo ""
        echo "  usb        → live USB webcam"
        echo "  csi        → live CSI camera (Jetson)"
        echo "  video      → pre-recorded video (loops)"
        echo "  save       → USB camera → output/camera_demo.mp4 (30s)"
        echo "  headless   → USB camera, no display, 10s"
        echo "  video-save → video → annotated output"
        echo "  bev-save   → video → BEV side-by-side output"
        echo "  all        → run all non-interactive tests"
        echo ""
        echo "  Add --bev to any mode:"
        echo "    ./scripts/test_camera.sh usb --bev"
        echo "    ./scripts/test_camera.sh video --bev"
        echo "    ./scripts/test_camera.sh save --bev"
        exit 1
        ;;
esac
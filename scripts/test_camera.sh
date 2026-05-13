#!/bin/bash
# test_camera.sh — Test all camera_capture modes
#
# Usage:
#   ./scripts/test_camera.sh [mode]
#
# Modes:
#   usb        — USB webcam live inference
#   csi        — CSI camera (Jetson nvarguscamerasrc)
#   video      — Pre-recorded video file (nuScenes clip)
#   save       — USB camera → save annotated output.mp4
#   headless   — USB camera, no display (SSH / no X11)
#   all        — Run all non-interactive tests sequentially
#
# Examples:
#   ./scripts/test_camera.sh usb
#   ./scripts/test_camera.sh video
#   ./scripts/test_camera.sh all

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

MODE="${1:-usb}"
ENGINE="weights/yolo26n_det_int8_raw.engine"
BIN="deployment/build/edgedrive"
THRESHOLD="0.3"

# ── Checks ────────────────────────────────────────────────────────────────────

check_bin() {
    if [ ! -f "$BIN" ]; then
        echo "Error: $BIN not found. Build first:"
        echo "  cd deployment && mkdir -p build && cd build && cmake .. && make -j4"
        exit 1
    fi
}

check_engine() {
    if [ ! -f "$ENGINE" ]; then
        echo "Error: $ENGINE not found. Export engine first:"
        echo "  ./scripts/build_engine.sh int8"
        exit 1
    fi
}

check_video() {
    shopt -s nullglob
    local files=()
    if [ -f "test_videos/sample.mp4" ]; then
        printf '%s\n' "test_videos/sample.mp4"
        return 0
    fi
    files=(test_videos/*.mp4)
    if [ ${#files[@]} -gt 0 ]; then
        printf '%s\n' "${files[0]}"
        return 0
    fi
    files=(*.mp4)
    if [ ${#files[@]} -gt 0 ]; then
        printf '%s\n' "${files[0]}"
        return 0
    fi
    return 1
}

check_display() {
    if [ -z "$DISPLAY" ] && [ -z "$WAYLAND_DISPLAY" ]; then
        echo "Warning: No display detected (DISPLAY not set)."
        echo "  → Use --headless mode or run with X11 forwarding: ssh -X user@jetson"
        echo "  → Or use: ./scripts/test_camera.sh headless"
        return 1
    fi
    return 0
}

check_camera() {
    local cam_id="${1:-0}"
    if [ ! -e "/dev/video${cam_id}" ]; then
        echo "Error: /dev/video${cam_id} not found."
        echo "  Check USB camera is connected: ls /dev/video*"
        exit 1
    fi
    echo "Camera /dev/video${cam_id} found."
}

check_bin
check_engine

# ── Test modes ────────────────────────────────────────────────────────────────

test_usb() {
    echo ""
    echo "================================================="
    echo "Test: USB Camera Live Inference"
    echo "Camera: /dev/video0 | Press 'q' or ESC to quit"
    echo "================================================="

    check_camera 0
    check_display || {
        echo "Falling back to headless mode."
        test_headless
        return
    }

    "$BIN" \
        --engine "$ENGINE" \
        --camera 0 \
        --threshold "$THRESHOLD"
}

test_csi() {
    echo ""
    echo "================================================="
    echo "Test: CSI Camera (nvarguscamerasrc)"
    echo "Sensor: 0 | Press 'q' or ESC to quit"
    echo "================================================="

    # Verify nvarguscamerasrc is available
    if ! gst-inspect-1.0 nvarguscamerasrc &>/dev/null; then
        echo "Error: nvarguscamerasrc not found."
        echo "  CSI camera requires JetPack with camera drivers."
        echo "  If using USB camera, use: ./scripts/test_camera.sh usb"
        exit 1
    fi

    check_display || {
        echo "Falling back to headless + save."
        "$BIN" \
            --engine "$ENGINE" \
            --csi 0 \
            --threshold "$THRESHOLD" \
            --no-display \
            --save-video output/csi_test.mp4
        return
    }

    "$BIN" \
        --engine "$ENGINE" \
        --csi 0 \
        --threshold "$THRESHOLD"
}

test_video() {
    echo ""
    echo "================================================="
    echo "Test: Pre-recorded Video Inference"
    echo "================================================="

    VIDEO=$(check_video)
    if [ -z "$VIDEO" ]; then
        echo "Error: No video file found."
        echo ""
        echo "Option A — use a nuScenes clip:"
        echo "  mkdir -p test_videos"
        echo "  # Copy any .mp4 driving clip to test_videos/"
        echo ""
        echo "Option B — create a clip from test_images:"
        echo "  python3 scripts/make_test_video.py"
        echo ""
        exit 1
    fi

    echo "Video: $VIDEO"
    echo "Press 'q' or ESC to quit. Video loops by default."

    check_display || {
        echo "No display — saving annotated output instead."
        mkdir -p output
        "$BIN" \
            --engine "$ENGINE" \
            --video "$VIDEO" \
            --threshold "$THRESHOLD" \
            --no-display \
            --no-loop \
            --save-video output/video_test.mp4
        echo "Saved: output/video_test.mp4"
        return
    }

    "$BIN" \
        --engine "$ENGINE" \
        --video "$VIDEO" \
        --threshold "$THRESHOLD"
}

test_save() {
    echo ""
    echo "================================================="
    echo "Test: Save Annotated Video (USB camera, 30s)"
    echo "Output: output/camera_demo.mp4"
    echo "================================================="

    check_camera 0
    mkdir -p output

    echo "Recording for 30 seconds... (Ctrl+C to stop early)"

    # Run in background, kill after 30s
    "$BIN" \
        --engine "$ENGINE" \
        --camera 0 \
        --threshold "$THRESHOLD" \
        --save-video output/camera_demo.mp4 \
        --no-display &

    BG_PID=$!
    sleep 30
    kill $BG_PID 2>/dev/null
    wait $BG_PID 2>/dev/null || true

    if [ -f output/camera_demo.mp4 ]; then
        SIZE=$(du -h output/camera_demo.mp4 | cut -f1)
        echo ""
        echo "✅ Saved: output/camera_demo.mp4 ($SIZE)"
        echo "   Play with: ffplay output/camera_demo.mp4"
        echo "   Or copy to laptop for review."
    else
        echo "❌ output/camera_demo.mp4 not created."
    fi
}

test_headless() {
    echo ""
    echo "================================================="
    echo "Test: Headless Inference (no display, 10s)"
    echo "Stats printed to console every 30 frames."
    echo "================================================="

    check_camera 0
    mkdir -p output

    echo "Running for 10 seconds..."
    timeout 10 "$BIN" \
        --engine "$ENGINE" \
        --camera 0 \
        --threshold "$THRESHOLD" \
        --no-display || true

    echo ""
    echo "✅ Headless test complete."
}

test_video_save() {
    echo ""
    echo "================================================="
    echo "Test: Video file → annotated output (demo reel)"
    echo "================================================="

    VIDEO=$(check_video)
    if [ -z "$VIDEO" ]; then
        echo "No video found. Create one first:"
        echo "  python3 scripts/make_test_video.py"
        exit 1
    fi

    mkdir -p output
    OUT="output/demo_$(basename "$VIDEO" .mp4)_annotated.mp4"

    echo "Input : $VIDEO"
    echo "Output: $OUT"
    echo "Running inference (no loop, no display)..."

    "$BIN" \
        --engine "$ENGINE" \
        --video "$VIDEO" \
        --threshold "$THRESHOLD" \
        --no-display \
        --no-loop \
        --save-video "$OUT"

    if [ -f "$OUT" ]; then
        SIZE=$(du -h "$OUT" | cut -f1)
        echo ""
        echo "✅ Done: $OUT ($SIZE)"
        echo "   This is your demo reel. Upload to GitHub or Google Drive."
    fi
}

test_all() {
    echo "Running all non-interactive tests..."
    echo ""
    test_headless
    test_video_save
    test_save
    echo ""
    echo "All non-interactive tests complete."
    echo "Run interactive tests manually:"
    echo "  ./scripts/test_camera.sh usb"
    echo "  ./scripts/test_camera.sh video"
}

# ── Main ──────────────────────────────────────────────────────────────────────

case "$MODE" in
    usb)        test_usb ;;
    csi)        test_csi ;;
    video)      test_video ;;
    save)       test_save ;;
    headless)   test_headless ;;
    video-save) test_video_save ;;
    all)        test_all ;;
    *)
        echo "Usage: $0 [usb|csi|video|save|headless|video-save|all]"
        echo ""
        echo "  usb        → live USB webcam (interactive)"
        echo "  csi        → live CSI camera (interactive, Jetson only)"
        echo "  video      → pre-recorded video file (interactive, loops)"
        echo "  save       → USB camera → output/camera_demo.mp4 (30s)"
        echo "  headless   → USB camera, no display, console stats"
        echo "  video-save → video file → annotated output (demo reel)"
        echo "  all        → run all non-interactive tests"
        exit 1
        ;;
esac
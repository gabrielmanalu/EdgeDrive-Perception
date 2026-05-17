#!/bin/bash
# run_ros2_bag_demo.sh — Full ROS2 pipeline: nuScenes bag → TRT inference
#
# Runs bag replay + camera_node in a single container.
# Equivalent of the full Phase 2 ROS2 demo.
#
# Usage:
#   ./scripts/run_ros2_bag_demo.sh                    # default scene0
#   ./scripts/run_ros2_bag_demo.sh nuscenes_scene0    # explicit bag name
#   BAG=nuscenes_all ./scripts/run_ros2_bag_demo.sh   # all scenes
#
# Prerequisites:
#   sudo jetson_clocks
#   Bag exists: bags/nuscenes_scene0/ (run scripts/nuscenes_to_ros2bag.py first)
#   Image built: docker build -t edgedrive-ros2:latest -f docker/Dockerfile.ros2 .

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

BAG="${1:-${BAG:-nuscenes_scene0}}"
ENGINE="${ENGINE:-yolo26n_det_int8_raw.engine}"
THRESH="${THRESH:-0.3}"

# ── Checks ────────────────────────────────────────────────────────────────────

if [ ! -d "bags/$BAG" ]; then
    echo "Error: bags/$BAG not found."
    echo "Generate it first:"
    echo "  python3 scripts/nuscenes_to_ros2bag.py \\"
    echo "      --dataroot /data/sets/nuscenes \\"
    echo "      --version v1.0-mini \\"
    echo "      --output bags/$BAG \\"
    echo "      --scene-idx 0"
    exit 1
fi

if ! docker image inspect edgedrive-ros2:latest &>/dev/null; then
    echo "Error: edgedrive-ros2:latest image not found."
    echo "Build it first:"
    echo "  docker build -t edgedrive-ros2:latest -f docker/Dockerfile.ros2 ."
    exit 1
fi

if [ ! -f "weights/$ENGINE" ]; then
    echo "Error: weights/$ENGINE not found."
    echo "Build engine first: ./scripts/build_engine.sh int8"
    exit 1
fi

# ── Info ──────────────────────────────────────────────────────────────────────

echo ""
echo "================================================="
echo "EdgeDrive ROS2 Bag Demo"
echo "================================================="
echo "  Bag     : bags/$BAG"
echo "  Engine  : weights/$ENGINE"
echo "  Threshold: $THRESH"
echo ""
echo "Press Ctrl+C to stop."
echo "================================================="
echo ""

# ── Run ───────────────────────────────────────────────────────────────────────

docker run --rm --runtime nvidia \
    --network host \
    --device /dev/nvmap \
    --device /dev/nvhost-ctrl-gpu \
    --device /dev/nvhost-gpu \
    --device /dev/nvhost-as-gpu \
    --device /dev/nvhost-tsg-gpu \
    -v "$REPO_ROOT/weights":/workspace/weights:ro \
    -v "$REPO_ROOT/bags":/workspace/bags:ro \
    edgedrive-ros2:latest \
    bash -c "
        source /opt/ros/humble/setup.bash
        source /workspace/ros2_ws/install/setup.bash

        # Start bag replay in background
        ros2 bag play /workspace/bags/$BAG --loop &
        BAG_PID=\$!

        # Wait for bag to start publishing
        sleep 3

        # Start camera node with topic remap
        ros2 run edgedrive_perception camera_node --ros-args \
            -p engine_path:=/workspace/weights/$ENGINE \
            -p score_threshold:=$THRESH \
            -p publish_viz:=false \
            -p publish_bev:=false \
            -r /camera/image_raw:=/nuscenes/camera/image_raw

        # Cleanup
        kill \$BAG_PID 2>/dev/null || true
    "
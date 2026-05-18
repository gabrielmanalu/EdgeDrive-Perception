#!/bin/bash
# run_ros2_rviz_demo.sh — Full ROS2 pipeline with RViz2 visualization
#
# Runs everything in one container:
#   - Static TF publisher (base_link → cam_front)
#   - nuScenes bag replay
#   - camera_node (TRT inference + MarkerArray + annotated image)
#   - RViz2
#
# Usage:
#   ./scripts/run_ros2_rviz_demo.sh
#   BAG=nuscenes_scene0 ./scripts/run_ros2_rviz_demo.sh
#
# In RViz2:
#   Fixed Frame  : base_link (already set)
#   Add → By topic → /detections/camera_markers → MarkerArray
#   Add → By topic → /camera/annotated          → Image
#
# Prerequisites:
#   sudo jetson_clocks
#   xhost +local:docker
#   Bag exists: bags/nuscenes_scene0/
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
    echo "      --output bags/$BAG --scene-idx 0"
    exit 1
fi

if ! docker image inspect edgedrive-ros2:latest &>/dev/null; then
    echo "Error: edgedrive-ros2:latest not found."
    echo "  docker build -t edgedrive-ros2:latest -f docker/Dockerfile.ros2 ."
    exit 1
fi

if [ ! -f "weights/$ENGINE" ]; then
    echo "Error: weights/$ENGINE not found."
    exit 1
fi

# ── X11 display check ─────────────────────────────────────────────────────────
if [ -z "$DISPLAY" ]; then
    echo "Warning: DISPLAY not set. RViz2 requires X11."
    echo "  Run: xhost +local:docker"
fi

echo ""
echo "================================================="
echo "EdgeDrive ROS2 RViz2 Demo"
echo "================================================="
echo "  Bag     : bags/$BAG"
echo "  Engine  : weights/$ENGINE"
echo ""
echo "RViz2 setup:"
echo "  Fixed Frame : base_link"
echo "  Add → /detections/camera_markers → MarkerArray"
echo "  Add → /camera/annotated          → Image"
echo ""
echo "Press Ctrl+C to stop."
echo "================================================="
echo ""

# ── Single container: TF + bag + camera_node + rviz2 ─────────────────────────

docker run --rm --runtime nvidia \
    --network host \
    --device /dev/nvmap \
    --device /dev/nvhost-ctrl-gpu \
    --device /dev/nvhost-gpu \
    --device /dev/nvhost-as-gpu \
    --device /dev/nvhost-tsg-gpu \
    -e DISPLAY="$DISPLAY" \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v "$REPO_ROOT/weights":/workspace/weights:ro \
    -v "$REPO_ROOT/bags":/workspace/bags:ro \
    edgedrive-ros2:latest \
    bash -c "
        source /opt/ros/humble/setup.bash
        source /workspace/ros2_ws/install/setup.bash

        echo '[1/4] Starting static TF publisher (base_link → cam_front)...'
        ros2 run tf2_ros static_transform_publisher \
            0 0 1.2 0 0 0 base_link cam_front &
        TF_PID=\$!

        echo '[2/4] Starting nuScenes bag replay...'
        ros2 bag play /workspace/bags/$BAG --loop &
        BAG_PID=\$!

        echo '[3/4] Waiting for bag to start...'
        sleep 3

        echo '[4/4] Starting camera_node (TRT inference + markers)...'
        ros2 run edgedrive_perception camera_node --ros-args \
            -p engine_path:=/workspace/weights/$ENGINE \
            -p score_threshold:=$THRESH \
            -p publish_viz:=true \
            -p publish_markers:=true \
            -p publish_bev:=false \
            -r /camera/image_raw:=/nuscenes/camera/image_raw &
        CAM_PID=\$!

        echo 'Opening RViz2...'
        echo ''
        echo 'In RViz2:'
        echo '  1. Fixed Frame is already set to base_link'
        echo '  2. Add → By topic → /detections/camera_markers → MarkerArray'
        echo '  3. Add → By topic → /camera/annotated → Image'
        echo ''
        sleep 2
        rviz2

        # Cleanup on RViz2 exit
        kill \$CAM_PID \$BAG_PID \$TF_PID 2>/dev/null || true
    "
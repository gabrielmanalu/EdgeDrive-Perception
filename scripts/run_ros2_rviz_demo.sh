#!/bin/bash
# run_ros2_rviz_demo.sh — Full ROS2 pipeline with RViz2 visualization
#
# Runs everything in one container:
#   - Static TF publisher (base_link → cam_front)
#   - nuScenes bag replay (camera + LiDAR)
#   - camera_node (TRT inference + green MarkerArray)
#   - lidar_detection_node (CUDA-PointPillars + blue MarkerArray)
#   - RViz2 (camera green + LiDAR blue cylinders side by side)
#
# Usage:
#   ./scripts/run_ros2_rviz_demo.sh
#   BAG=nuscenes_scene0 ./scripts/run_ros2_rviz_demo.sh
#
# Prerequisites:
#   sudo jetson_clocks
#   xhost +local:docker
#   scripts/setup_cuda_pointpillars.sh must have been run first

set -e

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

BAG="${1:-${BAG:-nuscenes_scene0}}"
ENGINE="${ENGINE:-yolo26n_det_int8_raw.engine}"
THRESH="${THRESH:-0.3}"

# ── Checks ────────────────────────────────────────────────────────────────────

if [ ! -d "bags/$BAG" ]; then
    echo "Error: bags/$BAG not found."
    echo "  python3 scripts/nuscenes_to_ros2bag.py --output bags/$BAG --scene-idx 0"
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

PP_PLAN="cuda-pointpillars/model/pointpillar.plan"
if [ ! -f "$PP_PLAN" ]; then
    echo "Warning: $PP_PLAN not found — lidar_detection_node will not start."
    echo "  Run: ./scripts/setup_cuda_pointpillars.sh"
    LIDAR_ENABLED=false
else
    LIDAR_ENABLED=true
fi

echo ""
echo "================================================="
echo "EdgeDrive ROS2 RViz2 Demo"
echo "================================================="
echo "  Bag     : bags/$BAG"
echo "  Engine  : weights/$ENGINE"
echo "  LiDAR   : $LIDAR_ENABLED"
echo ""
echo "RViz2:"
echo "  Green cylinders = camera detections"
echo "  Blue  cylinders = LiDAR detections"
echo ""
echo "Press Ctrl+C to stop."
echo "================================================="
echo ""

# ── Single container: TF + bag + camera_node + lidar_node + rviz2 ─────────────

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
    -v "$REPO_ROOT/ros2_ws/src":/workspace/ros2_ws/src:ro \
    -v "$REPO_ROOT/cuda-pointpillars/model":/workspace/cuda-pointpillars/model:ro \
    edgedrive-ros2:latest \
    bash -c "
        source /opt/ros/humble/setup.bash
        source /workspace/ros2_ws/install/setup.bash

        echo '[1/5] Static TF: base_link → cam_front'
        ros2 run tf2_ros static_transform_publisher \
            0 0 1.2 0 0 0 base_link cam_front &
        TF_PID=\$!

        echo '[2/5] Starting bag replay...'
        ros2 bag play /workspace/bags/$BAG --loop &
        BAG_PID=\$!

        echo '[3/5] Waiting for bag to start...'
        sleep 3

        echo '[4/5] Starting camera_node + lidar_detection_node...'
        ros2 run edgedrive_perception camera_node --ros-args \
            -p engine_path:=/workspace/weights/$ENGINE \
            -p score_threshold:=$THRESH \
            -p publish_viz:=true \
            -p publish_markers:=true \
            -p publish_bev:=false \
            -r /camera/image_raw:=/nuscenes/camera/image_raw &
        CAM_PID=\$!

        if [ -f /workspace/cuda-pointpillars/model/pointpillar.plan ]; then
            ros2 run edgedrive_perception lidar_detection_node --ros-args \
                -p engine_path:=/workspace/cuda-pointpillars/model/pointpillar_fpn3.plan \
                -p score_threshold:=0.2 \
                -p publish_markers:=true &
            LIDAR_PID=\$!
        else
            echo 'Warning: pointpillar.plan not found — skipping lidar_detection_node'
            LIDAR_PID=''
        fi

        # Start fusion node with nuScenes CAM_FRONT intrinsics
        ros2 run edgedrive_perception fusion_node --ros-args \
            -p match_threshold:=5.0 \
            -p sync_tolerance:=2.0 \
            -p camera_fx:=1266.417 \
            -p camera_fy:=1266.417 \
            -p camera_cx:=816.267 \
            -p camera_cy:=491.507 \
            -p camera_height:=1.5 &
        FUSION_PID=\$!

        echo '[5/5] Opening RViz2...'
        sleep 2
        rviz2 -d /workspace/ros2_ws/src/edgedrive_perception/config/edgedrive.rviz

        kill \$CAM_PID \$LIDAR_PID \$FUSION_PID \$BAG_PID \$TF_PID 2>/dev/null || true
    "
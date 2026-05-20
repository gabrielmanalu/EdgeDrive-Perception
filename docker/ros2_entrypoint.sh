#!/bin/bash
# ros2_entrypoint.sh — Source ROS2 + workspace on container start

set -e

source /opt/ros/humble/setup.bash

# Source built workspace if it exists
if [ -f /workspace/ros2_ws/install/setup.bash ]; then
    source /workspace/ros2_ws/install/setup.bash
fi

# Add CUDA-PointPillars library to path for lidar_detection_node
export LD_LIBRARY_PATH=/workspace/cuda-pointpillars/build:${LD_LIBRARY_PATH}

exec "$@"
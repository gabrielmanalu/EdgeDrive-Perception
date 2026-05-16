#!/bin/bash
# test_ros2_camera.sh — Test ROS2 camera_node with a synthetic image
#
# Usage:
#   ./scripts/test_ros2_camera.sh
#
# Prerequisites:
#   camera_node must be running in another terminal:
#   docker compose -f docker/docker-compose.ros2.yml run --rm camera-node

set -e

CONTAINER=$(docker ps -q -f name=camera-node | head -1)

if [ -z "$CONTAINER" ]; then
    echo "Error: camera-node container not running."
    echo "Start it first:"
    echo "  docker compose -f docker/docker-compose.ros2.yml run --rm camera-node"
    exit 1
fi

echo "Found camera-node container: $CONTAINER"
echo "Publishing 20 synthetic frames to /camera/image_raw..."
echo ""

docker exec "$CONTAINER" bash -c \
"source /opt/ros/humble/setup.bash && \
 source /workspace/ros2_ws/install/setup.bash && \
 python3 -c \"
import rclpy, time
import numpy as np
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

rclpy.init()
node = Node('test_image_pub')
qos = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    durability=DurabilityPolicy.VOLATILE,
    depth=10
)
pub = node.create_publisher(Image, '/camera/image_raw', qos)
bridge = CvBridge()

# Synthetic 1280x720 image with some structure
img = np.zeros((720, 1280, 3), dtype='uint8')
img[200:500, 300:600] = [200, 200, 200]  # grey rectangle (simulates car)
img[400:600, 800:1100] = [150, 150, 150]  # another rectangle
msg = bridge.cv2_to_imgmsg(img, 'bgr8')

time.sleep(1)
subs = pub.get_subscription_count()
print(f'Subscribers connected: {subs}')

# Wait until subscriber is discovered (max 10 seconds)
timeout = 10
waited = 0
while pub.get_subscription_count() == 0 and waited < timeout:
    print(f'Waiting for subscriber... ({waited}s)')
    time.sleep(0.5)
    waited += 0.5

if pub.get_subscription_count() == 0:
    print('Warning: no subscribers found after 10s. Is camera_node running?')
else:
    print(f'Subscriber connected after {waited}s')

for i in range(20):
    msg.header.stamp = node.get_clock().now().to_msg()
    pub.publish(msg)
    print(f'Published frame {i+1}/20')
    time.sleep(0.5)

print('Done. Check camera-node terminal for FPS and detection output.')
node.destroy_node()
rclpy.shutdown()
\""
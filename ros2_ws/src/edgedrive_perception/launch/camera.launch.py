"""
camera.launch.py — Launch YOLO26n camera detection node

Usage:
    ros2 launch edgedrive_perception camera.launch.py
    ros2 launch edgedrive_perception camera.launch.py engine:=/path/to/engine.engine
    ros2 launch edgedrive_perception camera.launch.py publish_bev:=true
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([

        # ── Arguments ─────────────────────────────────────────────────────────
        DeclareLaunchArgument(
            'engine',
            default_value='/workspace/weights/yolo26n_det_int8_raw.engine',
            description='Path to TensorRT engine file'
        ),
        DeclareLaunchArgument(
            'threshold',
            default_value='0.3',
            description='Detection score threshold'
        ),
        DeclareLaunchArgument(
            'publish_viz',
            default_value='true',
            description='Publish annotated image to /camera/annotated'
        ),
        DeclareLaunchArgument(
            'publish_bev',
            default_value='false',
            description='Publish BEV image to /camera/bev'
        ),
        DeclareLaunchArgument(
            'camera_height',
            default_value='1.2',
            description='Camera height above ground in meters'
        ),
        DeclareLaunchArgument(
            'image_topic',
            default_value='/camera/image_raw',
            description='Input image topic'
        ),

        # ── Camera node ────────────────────────────────────────────────────────
        Node(
            package='edgedrive_perception',
            executable='camera_node',
            name='camera_node',
            output='screen',
            parameters=[{
                'engine_path':      LaunchConfiguration('engine'),
                'score_threshold':  LaunchConfiguration('threshold'),
                'publish_viz':      LaunchConfiguration('publish_viz'),
                'publish_bev':      LaunchConfiguration('publish_bev'),
                'camera_height':    LaunchConfiguration('camera_height'),
            }],
            remappings=[
                ('/camera/image_raw', LaunchConfiguration('image_topic')),
            ]
        ),
    ])
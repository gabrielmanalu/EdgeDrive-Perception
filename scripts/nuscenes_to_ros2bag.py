#!/usr/bin/env python3
"""
nuscenes_to_ros2bag.py — Convert nuScenes Mini to ROS2 bag

Converts nuScenes camera + LiDAR data to a .db3 ROS2 bag file
for replay with the edgedrive_perception ROS2 nodes.

Topics published:
  /nuscenes/camera/image_raw    (sensor_msgs/Image)
  /nuscenes/lidar/pointcloud    (sensor_msgs/PointCloud2)

Usage:
  python3 scripts/nuscenes_to_ros2bag.py \
      --dataroot /data/sets/nuscenes \
      --version v1.0-mini \
      --output bags/nuscenes_scene0.db3 \
      --scene-idx 0

  python3 scripts/nuscenes_to_ros2bag.py --all-scenes \
      --output bags/nuscenes_all.db3

Requirements:
  pip3 install rosbags nuscenes-devkit
"""

import argparse
import os
import sys
import numpy as np
import cv2
from pathlib import Path


def check_dependencies():
    missing = []
    try:
        from nuscenes.nuscenes import NuScenes
    except ImportError:
        missing.append("nuscenes-devkit")
    try:
        from rosbags.rosbag2 import Writer
    except ImportError:
        missing.append("rosbags")
    if missing:
        print(f"Missing: {', '.join(missing)}")
        print(f"Install: pip3 install {' '.join(missing)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataroot", default="/data/sets/nuscenes")
    parser.add_argument("--version",  default="v1.0-mini")
    parser.add_argument("--output",   default="bags/nuscenes_scene0.db3")
    parser.add_argument("--scene-idx", type=int, default=0)
    parser.add_argument("--camera",   default="CAM_FRONT")
    parser.add_argument("--all-scenes", action="store_true")
    args = parser.parse_args()

    check_dependencies()

    from nuscenes.nuscenes import NuScenes
    from rosbags.rosbag2 import Writer
    from rosbags.typesys import Stores, get_typestore

    typestore   = get_typestore(Stores.ROS2_HUMBLE)
    Image       = typestore.types['sensor_msgs/msg/Image']
    PointCloud2 = typestore.types['sensor_msgs/msg/PointCloud2']
    PointField  = typestore.types['sensor_msgs/msg/PointField']
    Header      = typestore.types['std_msgs/msg/Header']
    Time        = typestore.types['builtin_interfaces/msg/Time']

    print(f"Loading nuScenes {args.version} from {args.dataroot}...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)

    scenes = nusc.scene if args.all_scenes else [nusc.scene[args.scene_idx]]
    print(f"Converting {len(scenes)} scene(s)...")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with Writer(str(out_path), version=8) as writer:
        cam_conn = writer.add_connection(
            "/nuscenes/camera/image_raw",
            Image.__msgtype__, typestore=typestore)
        lidar_conn = writer.add_connection(
            "/nuscenes/lidar/pointcloud",
            PointCloud2.__msgtype__, typestore=typestore)

        total = 0

        for scene in scenes:
            print(f"\nScene: {scene['name']} ({scene['nbr_samples']} samples)")
            sample_token = scene['first_sample_token']

            while sample_token:
                sample = nusc.get('sample', sample_token)
                ts_ns  = sample['timestamp'] * 1000  # us → ns

                sec  = int(ts_ns // 1_000_000_000)
                nsec = int(ts_ns %  1_000_000_000)

                # ── Camera ────────────────────────────────────────────────────
                if args.camera in sample['data']:
                    cam_data = nusc.get('sample_data', sample['data'][args.camera])
                    img_path = os.path.join(args.dataroot, cam_data['filename'])
                    img = cv2.imread(img_path)

                    if img is not None:
                        h, w = img.shape[:2]
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                        header = Header(
                            stamp=Time(sec=sec, nanosec=nsec),
                            frame_id="cam_front")

                        ros_img = Image(
                            header=header,
                            height=h, width=w,
                            encoding="rgb8",
                            is_bigendian=False,
                            step=w * 3,
                            data=img_rgb.flatten().astype(np.uint8))

                        writer.write(
                            cam_conn, ts_ns,
                            typestore.serialize_cdr(ros_img, Image.__msgtype__))

                # ── LiDAR ─────────────────────────────────────────────────────
                if 'LIDAR_TOP' in sample['data']:
                    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                    pcd_path = os.path.join(args.dataroot, lidar_data['filename'])

                    if os.path.exists(pcd_path):
                        points = np.fromfile(pcd_path, dtype=np.float32).reshape(-1, 5)

                        header = Header(
                            stamp=Time(sec=sec, nanosec=nsec),
                            frame_id="lidar_top")

                        fields = [
                            PointField(name="x",         offset=0,  datatype=7, count=1),
                            PointField(name="y",         offset=4,  datatype=7, count=1),
                            PointField(name="z",         offset=8,  datatype=7, count=1),
                            PointField(name="intensity", offset=12, datatype=7, count=1),
                            PointField(name="ring",      offset=16, datatype=7, count=1),
                        ]

                        ros_pcd = PointCloud2(
                            header=header,
                            height=1,
                            width=points.shape[0],
                            fields=fields,
                            is_bigendian=False,
                            point_step=20,
                            row_step=points.shape[0] * 20,
                            data=points.flatten().view(np.uint8),
                            is_dense=True)

                        writer.write(
                            lidar_conn, ts_ns,
                            typestore.serialize_cdr(ros_pcd, PointCloud2.__msgtype__))

                total += 1
                sample_token = sample['next']
                print(f"  Frame {total}", end='\r')

    print(f"\n✅ Done: {total} frames → {args.output}")
    print(f"   Topics:")
    print(f"     /nuscenes/camera/image_raw")
    print(f"     /nuscenes/lidar/pointcloud")
    print(f"\nReplay:")
    print(f"   docker compose -f docker/docker-compose.ros2.yml run --rm bag-replay")


if __name__ == "__main__":
    main()
from asyncio import base_subprocess
from email.mime import base

from sqlalchemy import true
from launch import LaunchDescription
from launch_ros.actions import Node
import os
from ament_index_python.packages import get_package_share_directory
import launch_ros
from pathlib import Path


def generate_launch_description():
    base_path = os.path.realpath(get_package_share_directory("camera_lidar_fusion"))

    return LaunchDescription(
        [
            Node(
                package="camera_lidar_fusion",
                executable="camera_lidar_fusion_node",
                name="camera_lidar_fusion_node",
                output="screen",
                emulate_tty=True,
                parameters=[],
            ),
        ]
    )

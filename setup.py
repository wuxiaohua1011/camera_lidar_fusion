from setuptools import setup
import os
from glob import glob

package_name = "camera_lidar_fusion"

setup(
    name=package_name,
    version="0.0.0",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name), glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="roar",
    maintainer_email="wuxiaohua1011@berkeley.edu",
    description="TODO: Package description",
    license="TODO: License declaration",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "camera_lidar_fusion_node = camera_lidar_fusion.camera_lidar_fusion_node:main"
        ],
    },
)

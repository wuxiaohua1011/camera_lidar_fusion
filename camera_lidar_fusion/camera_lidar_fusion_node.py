#!/usr/bin/env python3
import rclpy
import rclpy.node
import sys
import cv2
import numpy as np
from pathlib import Path
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import message_filters
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from tf2_ros.transform_listener import TransformListener
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException

import tf_transformations as tr
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Transform
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Vector3
from matplotlib import cm
import numpy as np

VIRIDIS = np.array(cm.get_cmap("viridis").colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])


def pose_to_pq(msg):
    """Convert a C{geometry_msgs/Pose} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    p = np.array([msg.position.x, msg.position.y, msg.position.z])
    q = np.array(
        [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
    )
    return p, q


def pose_stamped_to_pq(msg):
    """Convert a C{geometry_msgs/PoseStamped} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    return pose_to_pq(msg.pose)


def transform_to_pq(msg):
    """Convert a C{geometry_msgs/Transform} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    p = np.array([msg.translation.x, msg.translation.y, msg.translation.z])
    q = np.array([msg.rotation.x, msg.rotation.y, msg.rotation.z, msg.rotation.w])
    return p, q


def transform_stamped_to_pq(msg):
    """Convert a C{geometry_msgs/TransformStamped} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    return transform_to_pq(msg.transform)


def msg_to_se3(msg):
    """Conversion from geometric ROS messages into SE(3)

    @param msg: Message to transform. Acceptable types - C{geometry_msgs/Pose}, C{geometry_msgs/PoseStamped},
    C{geometry_msgs/Transform}, or C{geometry_msgs/TransformStamped}
    @return: a 4x4 SE(3) matrix as a numpy array
    @note: Throws TypeError if we receive an incorrect type.
    """
    if isinstance(msg, Pose):
        p, q = pose_to_pq(msg)
    elif isinstance(msg, PoseStamped):
        p, q = pose_stamped_to_pq(msg)
    elif isinstance(msg, Transform):
        p, q = transform_to_pq(msg)
    elif isinstance(msg, TransformStamped):
        p, q = transform_stamped_to_pq(msg)
    else:
        raise TypeError("Invalid type for conversion to SE(3)")
    norm = np.linalg.norm(q)
    if np.abs(norm - 1.0) > 1e-3:
        raise ValueError(
            "Received un-normalized quaternion (q = {0:s} ||q|| = {1:3.6f})".format(
                str(q), np.linalg.norm(q)
            )
        )
    elif np.abs(norm - 1.0) > 1e-6:
        q = q / norm
    g = tr.quaternion_matrix(q)
    g[0:3, -1] = p
    return g


class CameraLidarFusionNode(rclpy.node.Node):
    def __init__(self):
        super().__init__("test_node")
        self.front_left_img = message_filters.Subscriber(
            self, Image, "/carla/ego_vehicle/front_left_rgb/image"
        )
        self.front_left_cam_info = message_filters.Subscriber(
            self, Image, "/carla/ego_vehicle/front_left_rgb/camera_info"
        )
        self.center_lidar = message_filters.Subscriber(
            self, PointCloud2, "/carla/ego_vehicle/center_lidar"
        )
        queue_size = 30
        self.ts = message_filters.TimeSynchronizer(
            [self.front_left_img, self.center_lidar],
            queue_size,
        )
        self.ts.registerCallback(self.callback)

        self.subscription = self.create_subscription(
            CameraInfo,
            "/carla/ego_vehicle/front_left_rgb/camera_info",
            self.camera_info_callback,
            10,
        )
        self.subscription  # prevent unused variable warning
        self.intrinsics = np.zeros(shape=(3, 3))
        self.image_w = 762
        self.image_h = 386
        self.bridge = CvBridge()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.to_frame_rel = "ego_vehicle/front_left_rgb"
        self.from_frame_rel = "ego_vehicle/center_lidar"
        # self.vis = o3d.visualization.Visualizer()
        # self.vis.create_window(width=500, height=500)
        # self.pcd = o3d.geometry.PointCloud()
        # self.coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        # self.points_added = False

    def camera_info_callback(self, msg):
        self.intrinsics = np.reshape(msg.k, newshape=(3, 3))
        self.image_w = msg.width
        self.image_h = msg.height

    def callback(self, left_cam_msg, center_lidar_pcl_msg):
        left_img = self.bridge.imgmsg_to_cv2(left_cam_msg)

        pcd_as_numpy_array = np.array(list(read_points(center_lidar_pcl_msg)))
        point_cloud = pcd_as_numpy_array[:, :3]
        intensity = pcd_as_numpy_array[:, 3]
        # self.o3d_pcd = o3d.geometry.PointCloud(
        #     o3d.utility.Vector3dVector(pcd_as_numpy_array)
        # )

        try:
            now = rclpy.time.Time()
            trans_msg = self.tf_buffer.lookup_transform(
                self.to_frame_rel, self.from_frame_rel, now
            )
            lidar2cam = msg_to_se3(trans_msg)
        except TransformException as ex:
            self.get_logger().info(
                f"Could not transform {self.to_frame_rel} to {self.from_frame_rel}: {ex}"
            )
            return

        # Point cloud in lidar sensor space array of shape (3, p_cloud_size).
        local_lidar_points = np.array(point_cloud).T

        # Add an extra 1.0 at the end of each 3d point so it becomes of
        # shape (4, p_cloud_size) and it can be multiplied by a (4, 4) matrix.
        local_lidar_points = np.r_[
            local_lidar_points, [np.ones(local_lidar_points.shape[1])]
        ]

        # now we tranform the points from lidar to camera
        sensor_points = np.dot(lidar2cam, local_lidar_points)
        point_in_camera_coords = np.array(
            [sensor_points[0], sensor_points[1], sensor_points[2]]
        )

        points_2d = np.dot(self.intrinsics, point_in_camera_coords)

        # Remember to normalize the x, y values by the 3rd value.
        points_2d = np.array(
            [
                points_2d[0, :] / points_2d[2, :],
                points_2d[1, :] / points_2d[2, :],
                points_2d[2, :],
            ]
        )
        # At this point, points_2d[0, :] contains all the x and points_2d[1, :]
        # contains all the y values of our points. In order to properly
        # visualize everything on a screen, the points that are out of the screen
        # must be discarted, the same with points behind the camera projection plane.
        points_2d = points_2d.T
        intensity = intensity.T
        points_in_canvas_mask = (
            (points_2d[:, 0] > 0.0)
            & (points_2d[:, 0] < self.image_w)
            & (points_2d[:, 1] > 0.0)
            & (points_2d[:, 1] < self.image_h)
            & (points_2d[:, 2] > 0.0)
        )
        points_2d = points_2d[points_in_canvas_mask]
        intensity = intensity[points_in_canvas_mask]

        # Extract the screen coords (uv) as integers.
        u_coord = points_2d[:, 0].astype(np.int)
        v_coord = points_2d[:, 1].astype(np.int)

        # Since at the time of the creation of this script, the intensity function
        # is returning high values, these are adjusted to be nicely visualized.
        intensity = 4 * intensity - 3
        color_map = (
            np.array(
                [
                    np.interp(intensity, VID_RANGE, VIRIDIS[:, 0]) * 255.0,
                    np.interp(intensity, VID_RANGE, VIRIDIS[:, 1]) * 255.0,
                    np.interp(intensity, VID_RANGE, VIRIDIS[:, 2]) * 255.0,
                ]
            )
            .astype(np.int)
            .T
        )
        s = 1
        im_array = np.copy(left_img)[:, :, :3]
        # im_array[v_coord, u_coord] = color_map
        # Draw the 2d points on the image as squares of extent args.dot_extent.
        for i in range(len(points_2d)):
            # I'm not a NumPy expert and I don't know how to set bigger dots
            # without using this loop, so if anyone has a better solution,
            # make sure to update this script. Meanwhile, it's fast enough :)
            im_array[
                v_coord[i] - s : v_coord[i] + s,
                u_coord[i] - s : u_coord[i] + s,
            ] = color_map[i]
        cv2.imshow("img", im_array)
        cv2.waitKey(1)

    def non_blocking_pcd_visualization(
        self,
        pcd: o3d.geometry.PointCloud,
        should_center=False,
        should_show_axis=False,
        axis_size: float = 1,
    ):
        """
        Real time point cloud visualization.
        Args:
            pcd: point cloud to be visualized
            should_center: true to always center the point cloud
            should_show_axis: true to show axis
            axis_size: adjust axis size
        Returns:
            None
        """
        points = np.asarray(pcd.points)
        colors = np.asarray(pcd.colors)
        if should_center:
            points = points - np.mean(points, axis=0)

        if self.points_added is False:
            self.pcd = o3d.geometry.PointCloud()
            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.pcd.colors = o3d.utility.Vector3dVector(colors)

            if should_show_axis:
                self.coordinate_frame = (
                    o3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=axis_size, origin=np.mean(points, axis=0)
                    )
                )
                self.vis.add_geometry(self.coordinate_frame)
            self.vis.add_geometry(self.pcd)
            self.points_added = True
        else:
            # print(np.shape(np.vstack((np.asarray(self.pcd.points), points))))
            self.pcd.points = o3d.utility.Vector3dVector(points)
            self.pcd.colors = o3d.utility.Vector3dVector(colors)
            if should_show_axis:
                self.coordinate_frame = (
                    o3d.geometry.TriangleMesh.create_coordinate_frame(
                        size=axis_size, origin=np.mean(points, axis=0)
                    )
                )
                self.vis.update_geometry(self.coordinate_frame)
            self.vis.update_geometry(self.pcd)

        self.vis.poll_events()
        self.vis.update_renderer()


def main(args=None):
    rclpy.init()
    node = CameraLidarFusionNode()
    rclpy.spin(node)


## The code below is "ported" from
# https://github.com/ros/common_msgs/tree/noetic-devel/sensor_msgs/src/sensor_msgs
# I'll make an official port and PR to this repo later:
# https://github.com/ros2/common_interfaces
import sys
from collections import namedtuple
import ctypes
import math
import struct
from sensor_msgs.msg import PointCloud2, PointField

_DATATYPES = {}
_DATATYPES[PointField.INT8] = ("b", 1)
_DATATYPES[PointField.UINT8] = ("B", 1)
_DATATYPES[PointField.INT16] = ("h", 2)
_DATATYPES[PointField.UINT16] = ("H", 2)
_DATATYPES[PointField.INT32] = ("i", 4)
_DATATYPES[PointField.UINT32] = ("I", 4)
_DATATYPES[PointField.FLOAT32] = ("f", 4)
_DATATYPES[PointField.FLOAT64] = ("d", 8)


def read_points(cloud, field_names=None, skip_nans=False, uvs=[]):
    """
    Read points from a L{sensor_msgs.PointCloud2} message.
    @param cloud: The point cloud to read from.
    @type  cloud: L{sensor_msgs.PointCloud2}
    @param field_names: The names of fields to read. If None, read all fields. [default: None]
    @type  field_names: iterable
    @param skip_nans: If True, then don't return any point with a NaN value.
    @type  skip_nans: bool [default: False]
    @param uvs: If specified, then only return the points at the given coordinates. [default: empty list]
    @type  uvs: iterable
    @return: Generator which yields a list of values for each point.
    @rtype:  generator
    """
    assert isinstance(cloud, PointCloud2), "cloud is not a sensor_msgs.msg.PointCloud2"
    fmt = _get_struct_fmt(cloud.is_bigendian, cloud.fields, field_names)
    width, height, point_step, row_step, data, isnan = (
        cloud.width,
        cloud.height,
        cloud.point_step,
        cloud.row_step,
        cloud.data,
        math.isnan,
    )
    unpack_from = struct.Struct(fmt).unpack_from

    if skip_nans:
        if uvs:
            for u, v in uvs:
                p = unpack_from(data, (row_step * v) + (point_step * u))
                has_nan = False
                for pv in p:
                    if isnan(pv):
                        has_nan = True
                        break
                if not has_nan:
                    yield p
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    p = unpack_from(data, offset)
                    has_nan = False
                    for pv in p:
                        if isnan(pv):
                            has_nan = True
                            break
                    if not has_nan:
                        yield p
                    offset += point_step
    else:
        if uvs:
            for u, v in uvs:
                yield unpack_from(data, (row_step * v) + (point_step * u))
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    yield unpack_from(data, offset)
                    offset += point_step


def _get_struct_fmt(is_bigendian, fields, field_names=None):
    fmt = ">" if is_bigendian else "<"

    offset = 0
    for field in (
        f
        for f in sorted(fields, key=lambda f: f.offset)
        if field_names is None or f.name in field_names
    ):
        if offset < field.offset:
            fmt += "x" * (field.offset - offset)
            offset = field.offset
        if field.datatype not in _DATATYPES:
            print(
                "Skipping unknown PointField datatype [%d]" % field.datatype,
                file=sys.stderr,
            )
        else:
            datatype_fmt, datatype_length = _DATATYPES[field.datatype]
            fmt += field.count * datatype_fmt
            offset += field.count * datatype_length

    return fmt


if __name__ == "__main__":
    main()

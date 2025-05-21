import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header

import cv2
import numpy as np
import yaml
import os
from scipy.spatial.transform import Rotation as R_scipy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import TransformStamped
import threading
from .aruco_config import ArucoConfig

class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector')

        self.bridge = CvBridge()
        self.config = ArucoConfig()
        
        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/image/compressed',
            self.image_callback,
            10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)


        self.detected_marker_map = {}  
        threading.Thread(target=self.key_listener, daemon=True).start()
        self.save = False

        self.image_pub = self.create_publisher(Image, '/aruco_detector/detected_image', 10)
        self.marker_pose_pub = self.create_publisher(PoseStamped, '/aruco_detector/marker_pose', 10)
        
    def key_listener(self):
        import sys, termios, tty
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while True:
                key = sys.stdin.read(1)
                if key.lower() == 'q':
                    rclpy.shutdown()
                    break
                if key.lower() == 's':
                    self.save = True
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def get_transform(self):
        try:
            trans = self.tf_buffer.lookup_transform('odom', 'base_footprint', rclpy.time.Time())
            T = np.eye(4)
            T[0:3, 3] = [
                trans.transform.translation.x,
                trans.transform.translation.y,
                trans.transform.translation.z
            ]
            q = trans.transform.rotation
            r = R_scipy.from_quat([q.x, q.y, q.z, q.w])
            T[0:3, 0:3] = r.as_matrix()
            return T
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            return None


    def image_callback(self, msg):
        T_map_base = self.get_transform()
        if T_map_base is None:
            print("not T_map_base")
            return

        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if cv_image is None:
            print("not cv_image")
            return

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.config.aruco_dict, parameters=self.config.aruco_params)
        if ids is None:
            return

        gray_float = np.float32(gray) 
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )
        for i in range(len(corners)):
            cv2.cornerSubPix(
                gray_float,
                corners[i], 
                winSize=(5, 5),
                zeroZone=(-1, -1),
                criteria=criteria
            )
        cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)

        for i, marker_id in enumerate(ids.flatten()):
            image_points = corners[i]
            if marker_id not in [3,4,5]:
                reorder = [2, 3, 0, 1]
                image_points = image_points[:, reorder, :]
            
            success, rvec, tvec, inliersinliers = cv2.solvePnPRansac(
                self.config.objp,
                image_points,
                self.config.camera_matrix,
                self.config.dist_coeffs,
                # flags=cv2.SOLVEPNP_ITERATIVE
            )

            if not success:
                continue
            cv2.drawFrameAxes(cv_image, self.config.camera_matrix, self.config.dist_coeffs, rvec, tvec, self.config.marker_length * 0.5)
           
            R, _ = cv2.Rodrigues(rvec)
            T_camera_marker = np.eye(4)
            T_camera_marker[0:3, 0:3] = R
            T_camera_marker[0:3, 3] = tvec.flatten()

            T_map_marker = T_map_base @ self.config.T_camera_base @ T_camera_marker

            pos = T_map_marker[0:3, 3]
            yaw = np.arctan2(T_map_marker[1, 0], T_map_marker[0, 0])

            print(f"[ArUco {marker_id}] x={pos[0]:.3f}, y={pos[1]:.3f}, theta={yaw:.3f} rad")

            if self.save:
                self.detected_marker_map[int(marker_id)] = {
                    'x': float(pos[0]),
                    'y': float(pos[1]),
                    'theta': float(yaw)
                }
                self.save = False

            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = "map"
            pose_msg.pose.position.x = pos[0]
            pose_msg.pose.position.y = pos[1]
            pose_msg.pose.position.z = 0.0

            qz = np.sin(yaw / 2.0)
            qw = np.cos(yaw / 2.0)
            pose_msg.pose.orientation.z = qz
            pose_msg.pose.orientation.w = qw

            self.marker_pose_pub.publish(pose_msg)

        img_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        self.image_pub.publish(img_msg)

    def destroy_node(self):
        super().destroy_node()
        print("\n========= ArUco  =========")
        print(yaml.dump(self.detected_marker_map, sort_keys=True, default_flow_style=False))
        print("===========================================")

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

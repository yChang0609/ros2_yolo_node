import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import PoseWithCovarianceStamped
from cv_bridge import CvBridge
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

import cv2
import numpy as np
import yaml
import os


class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector')

        self.bridge = CvBridge()
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.aruco_params.adaptiveThreshWinSizeMin = 5
        self.aruco_params.adaptiveThreshWinSizeMax = 15
        self.aruco_params.adaptiveThreshWinSizeStep = 5

        # self.tf_broadcaster = TransformBroadcaster(self)
        self.image_pub = self.create_publisher(Image, '/aruco_detector/detected_image', 10)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/aruco_detector/pose', 10)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)

        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/image/compressed',
            self.image_callback,
            10)
        
        self.camera_matrix = np.array([
                [387.96929, 0.0, 320.0],
                [0.0, 390.85762, 240.0],
                [0.0, 0.0, 1.0]
            ])
        self.dist_coeffs = np.zeros(5) #np.array([0.147622, -0.198939, -0.009281, 0.009981, 0.0])
        self.camera_info_received = True

        # 載入 ArUco map yaml (format: {id: {x, y, theta}})
        # map_path = os.path.join(os.path.dirname(__file__), 'aruco_map.yaml')
        aruco_path = '/workspaces/src/aruco_detector/config/aruco_location.yaml'
        with open(aruco_path, 'r') as f:
            self.marker_map = yaml.safe_load(f)
        print(self.marker_map)

        # yaml_path = os.path.join(os.path.dirname(__file__), 'map.yaml')
        map_path = '/workspaces/src/aruco_detector/map'
        yaml_path = yaml_path = os.path.join(map_path, 'map01.yaml')
        with open(yaml_path, 'r') as f:
            map_metadata = yaml.safe_load(f)
        image_path = yaml_path = os.path.join(map_path, map_metadata['image'])
        resolution = map_metadata['resolution']
        origin = map_metadata['origin']
        occupied_thresh = map_metadata.get('occupied_thresh', 0.65)

        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Can't load map: {image_path}")
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        img = cv2.flip(img, 0)
        h, w = img.shape

        data = []
        for row in img:
            for pixel in row:
                occ = 100 if pixel < occupied_thresh * 255 else 0
                data.append(occ)

        self.map_msg = OccupancyGrid()
        self.map_msg.header = Header()
        self.map_msg.info.resolution = resolution
        self.map_msg.info.width = w
        self.map_msg.info.height = h
        self.map_msg.info.origin.position.x = origin[0]
        self.map_msg.info.origin.position.y = origin[1]
        self.map_msg.info.origin.position.z = 0.0
        self.map_msg.info.origin.orientation.w = 1.0
        self.map_msg.data = data

        self.timer = self.create_timer(1.0, self.publish_map)

        R = np.array([
            [1,  0,  0],   # x unchanged
            [0, -1,  0],   # y flipped
            [0,  0, -1]    # z flipped
        ])

        self.T_camera_base = np.eye(4)
        self.T_camera_base[0:3, 0:3] = R
        self.T_camera_base[0:3, 3] = [0.05, 0.0, 0.20]  # 你量到的 camera 在 base_link 的位置

    def publish_map(self):
        self.map_msg.header.stamp = self.get_clock().now().to_msg()
        self.map_msg.header.frame_id = "map"  
        self.map_pub.publish(self.map_msg)

    def image_callback(self, msg):
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if cv_image is None:
            return

        corners, ids, _ = cv2.aruco.detectMarkers(cv_image, self.aruco_dict, parameters=self.aruco_params)
        if ids is None:
            return
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        gray = np.float32(gray)
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )
        for corner in corners:
            cv2.cornerSubPix(
                gray,
                corner,
                winSize=(5, 5),
                zeroZone=(-1, -1),
                criteria=criteria
            )
        cv2.aruco.drawDetectedMarkers(cv_image, corners, ids)

        # estimate pose
        marker_length = 0.36  # meter
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, self.camera_matrix, self.dist_coeffs)

        for i, marker_id in enumerate(ids.flatten()):
        
            if marker_id not in self.marker_map:
                continue
            cv2.drawFrameAxes(
                cv_image,
                self.camera_matrix,
                self.dist_coeffs,
                rvecs[i],
                tvecs[i],
                marker_length * 0.5  # 軸長度（可調整）
            )
            origin_3d = np.array([[0, 0, 0]], dtype=np.float32)
            imgpts, _ = cv2.projectPoints(origin_3d, rvecs[i], tvecs[i], self.camera_matrix, self.dist_coeffs)
            u, v = imgpts[0][0]
            cv2.circle(cv_image, (int(u), int(v)), 5, (255, 0, 255), -1)

            
            # Get transform from camera to marker
            t = tvecs[i].flatten()
            R, _ = cv2.Rodrigues(rvecs[i])
            T_camera_marker = np.eye(4)
            T_camera_marker[0:3, 0:3] = R
            T_camera_marker[0:3, 3] = t

            # Get marker pose in map frame
            R_align = np.array([
                [ 0,  0,  1],   # map x → camera -z
                [ 1,  0,  0],   # map y → camera x
                [ 0, -1,  0]    # map z → camera -y
            ])

            m = self.marker_map[marker_id]
            theta = m['theta']
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            R_map_marker = np.array([
                [cos_t, -sin_t, 0],
                [sin_t,  cos_t, 0],
                [0,      0,     1]
            ])
            t_map_marker = np.array([m['x'], m['y'], 0])
            T_map_marker = np.eye(4)
            T_map_marker[0:3, 0:3] = R_map_marker @ R_align
            T_map_marker[0:3, 3] = t_map_marker

            # 推算相機（在 map）的位置：T_map_camera = T_map_marker * T_marker_camera
       
            T_marker_camera = np.linalg.inv(T_camera_marker)
            T_map_camera = T_map_marker @ T_marker_camera
            T_map_base     = T_map_camera @ self.T_camera_base

            # 如果 camera 和 base_link 有相對偏移，進一步算 base_link
            # T_map_base = T_map_camera @ self.T_camera_base
            # print(T_map_base)
            pos = T_map_base[0:3, 3]
            yaw = np.arctan2(T_map_base[1, 0], T_map_base[0, 0])
            
            # print(f"[DEBUG] Robot 推測位置： x={pos[0]:.3f}, y={pos[1]:.3f}")

            # 發布 PoseWithCovarianceStamped
            print(f"{pos},{yaw}")
            pose_msg = PoseWithCovarianceStamped()
            pose_msg.header.frame_id = "map"
            pose_msg.pose.pose.position.x = pos[0]
            pose_msg.pose.pose.position.y = pos[1]
            pose_msg.pose.pose.position.z = 0.0

            q = self.yaw_to_quaternion(yaw)
            pose_msg.pose.pose.orientation.x = q[0]
            pose_msg.pose.pose.orientation.y = q[1]
            pose_msg.pose.pose.orientation.z = q[2]
            pose_msg.pose.pose.orientation.w = q[3]
            print("pose estimatation!!")
            self.pose_pub.publish(pose_msg)
            break  # 只用第一個已知的 marker

        img_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        self.image_pub.publish(img_msg)

        # t = TransformStamped()
        # t.header.stamp = self.get_clock().now().to_msg()
        # t.header.frame_id = "map"
        # t.child_frame_id = "base_link"
        # t.transform.translation.x = pos[0]
        # t.transform.translation.y = pos[1]
        # t.transform.translation.z = 0.0
        # t.transform.rotation.x = q[0]
        # t.transform.rotation.y = q[1]
        # t.transform.rotation.z = q[2]
        # t.transform.rotation.w = q[3]
        # self.tf_broadcaster.sendTransform(t)

    def yaw_to_quaternion(self, yaw):
        qx = 0.0
        qy = 0.0
        qz = np.sin(yaw / 2.0)
        qw = np.cos(yaw / 2.0)
        return (qx, qy, qz, qw)


def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
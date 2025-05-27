import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage, Image
from geometry_msgs.msg import PoseWithCovarianceStamped
from cv_bridge import CvBridge
from nav_msgs.msg import OccupancyGrid
from std_msgs.msg import Header
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped

from scipy.spatial.transform import Rotation as scipy_R
import cv2
import numpy as np
import yaml
import os

from .aruco_config import ArucoConfig

from interfaces_pkg.msg import ArucoMarker, ArucoMarkerConfig

class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector')

        self.bridge = CvBridge()
        self.config = ArucoConfig()

        self.tf_broadcaster = TransformBroadcaster(self)
        self.image_pub = self.create_publisher(Image, '/aruco_detector/detected_image', 10)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/aruco_detector/pose', 10)
        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)

        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/image/compressed',
            self.image_callback,
            10)
        
        self.config_sub = self.create_subscription(
            ArucoMarkerConfig,
            '/aruco_marker/config',
            self.config_callback,
            10
        )
        self.marker_config = None
        self.unflipped_ids = None

        # yaml_path = os.path.join(os.path.dirname(__file__), 'map.yaml')
        # map_path = '/workspaces/src/aruco_detector/map'
        # yaml_path = yaml_path = os.path.join(map_path, 'map01.yaml')
        # with open(yaml_path, 'r') as f:
        #     map_metadata = yaml.safe_load(f)
        # image_path = yaml_path = os.path.join(map_path, map_metadata['image'])
        # resolution = map_metadata['resolution']
        # origin = map_metadata['origin']
        # occupied_thresh = map_metadata.get('occupied_thresh', 0.65)

        # img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        # if img is None:
        #     raise RuntimeError(f"Can't load map: {image_path}")
        # if len(img.shape) == 3:
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # img = cv2.flip(img, 0)
        # h, w = img.shape

        # data = []
        # for row in img:
        #     for pixel in row:
        #         occ = 100 if pixel < occupied_thresh * 255 else 0
        #         data.append(occ)

        # self.map_msg = OccupancyGrid()
        # self.map_msg.header = Header()
        # self.map_msg.info.resolution = resolution
        # self.map_msg.info.width = w
        # self.map_msg.info.height = h
        # self.map_msg.info.origin.position.x = origin[0]
        # self.map_msg.info.origin.position.y = origin[1]
        # self.map_msg.info.origin.position.z = 0.0
        # self.map_msg.info.origin.orientation.w = 1.0
        # self.map_msg.data = data

        # self.timer = self.create_timer(1.0, self.publish_map)

    # def publish_map(self):
    #     self.map_msg.header.stamp = self.get_clock().now().to_msg()
    #     self.map_msg.header.frame_id = "map"  
    #     self.map_pub.publish(self.map_msg)

    def polygon_area(self, pts):
        pts = np.array(pts[0])
        return 0.5 * abs(np.dot(pts[:,0], np.roll(pts[:,1], 1)) - np.dot(pts[:,1], np.roll(pts[:,0], 1)))
    
    def config_callback(self, msg: ArucoMarkerConfig):
        self.marker_config = {m.id: {'x': m.x, 'y': m.y, 'theta': m.theta} for m in msg.markers}
        self.unflipped_ids = list(msg.unflipped_ids)

        self.get_logger().info('Received marker config from topic:')
        for mid, pose in self.marker_config.items():
            flipped = '' if mid in self.unflipped_ids else '(flipped)'
            self.get_logger().info(f'  ID {mid}: {pose} {flipped}')

    def image_callback(self, msg):
        if not self.marker_config:
            return
        
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if cv_image is None:
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
        
            if marker_id not in self.marker_config  or self.polygon_area(corners[i])<1500:
                continue
      
            image_points = corners[i]  # shape: (4, 2)
            if marker_id not in self.unflipped_ids:
                reorder = [2, 3, 0, 1]
                image_points = image_points[:, reorder, :]

            # success, rvec, tvec, inliers = cv2.solvePnPRansac(
            #     objp, image_points,
            #     self.camera_matrix,
            #     self.dist_coeffs,
            #     flags=cv2.SOLVEPNP_ITERATIVE,  # 或 cv2.SOLVEPNP_ITERATIVE 更穩
            #     reprojectionError=3.0,
            #     confidence=0.99,
            #     iterationsCount=100
            # )
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

            # Get transform from camera to marker
            R, _ = cv2.Rodrigues(rvec)
            T_camera_marker = np.eye(4)
            T_camera_marker[0:3, 0:3] = R
            T_camera_marker[0:3, 3] = tvec.flatten()


            m = self.marker_config[marker_id]
            theta = m['theta']
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            R_map_marker = np.array([
                [cos_t, -sin_t, 0],
                [sin_t,  cos_t, 0],
                [0,      0,     1]
            ])
            location_map_marker = np.array([m['x'], m['y'], 0.0], dtype=np.float32)
            T_map_marker = np.eye(4)
            T_map_marker[0:3, 0:3] = R_map_marker
            T_map_marker[0:3, 3] = location_map_marker

            T_camer_laser = np.eye(4)
            temp = np.array([
                [-1, 0, 0],
                [0, -1, 0],
                [0, 0, 1]
            ])
            T_camer_laser[0:3, 0:3] = temp

            T_marker_camera = np.linalg.inv(T_camera_marker)
            T_map_camera = T_map_marker @ self.config.T_align @ T_marker_camera
            T_map_base = T_map_camera  @ np.linalg.inv(self.config.T_camera_base)

            pos = T_map_base[0:3, 3]

            R_matrix = T_map_base[0:3, 0:3]
            q = scipy_R.from_matrix(R_matrix).as_quat()
   
            # PoseWithCovarianceStamped
            print(f"{pos}")
            pose_msg = PoseWithCovarianceStamped()
            pose_msg.header.frame_id = "map"
            pose_msg.pose.pose.position.x = pos[0]
            pose_msg.pose.pose.position.y = pos[1]
            pose_msg.pose.pose.position.z = 0.0

            # q = self.yaw_to_quaternion(yaw)
            pose_msg.pose.pose.orientation.x = q[0]
            pose_msg.pose.pose.orientation.y = q[1]
            pose_msg.pose.pose.orientation.z = q[2]
            pose_msg.pose.pose.orientation.w = q[3]
            print("pose estimatation!!")
            self.pose_pub.publish(pose_msg)
           
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = "map"
            t.child_frame_id = "base_link"
            t.transform.translation.x = pos[0]
            t.transform.translation.y = pos[1]
            t.transform.translation.z = 0.0
            t.transform.rotation.x = q[0]
            t.transform.rotation.y = q[1]
            t.transform.rotation.z = q[2]
            t.transform.rotation.w = q[3]
            self.tf_broadcaster.sendTransform(t)
            break  

        img_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        self.image_pub.publish(img_msg)



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
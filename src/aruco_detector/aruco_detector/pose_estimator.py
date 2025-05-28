## >> ROS2
import rclpy
from rclpy.node import Node

# >> Basic package
import cv2
import yaml
import threading
import numpy as np
from collections import deque
from cv_bridge import CvBridge

## >> ROS2 interfaces
from interfaces_pkg.msg import ArucoMarkerConfig
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Float32MultiArray, Bool
from geometry_msgs.msg import PoseWithCovarianceStamped
from scipy.spatial.transform import Rotation as scipy_R

# >> Self package
from aruco_detector.aruco_config import ArucoConfig


class ArucoDetectorNode(Node):
    def __init__(self):
        super().__init__('aruco_detector')
        self.debug = True
        self.bridge = CvBridge()
        self.config = ArucoConfig()

        self.image_pub = self.create_publisher(Image, '/aruco_detector/detected_image', 10)
        self.pose_pub = self.create_publisher(PoseWithCovarianceStamped, '/aruco_detector/pose', 10)
        self.debug_odom_pub = self.create_publisher(PoseWithCovarianceStamped, '/aruco_detector/deubg/odom/pose', 10)
        self.debug_marker_pub = self.create_publisher(PoseWithCovarianceStamped, '/aruco_detector/debug/marker/pose', 10)


        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/image/compressed',
            self.image_callback, 10)
        
        self.wheel_radius = 0.04  # m
        self.wheel_base = 0.23    # m

        self.odom_pose = np.array([0.0, 0.0, 0.0])  # x, y, theta
        self.last_time = self.get_clock().now().nanoseconds / 1e9

        self.rear_data = [0.0, 0.0]
        self.front_data = [0.0, 0.0]
        self.create_subscription(Float32MultiArray, "/car_C_rear_wheel", self.rear_callback, 10)
        self.create_subscription(Float32MultiArray, "/car_C_front_wheel", self.front_callback, 10)
        self.create_timer(0.05, self.update_odometry_timer)  

        self.change_scene_sub = self.create_subscription(
            Bool,
            "/change_scene",
            self.scene_reset_callback,
            10
        )

        self.test = True
        if(self.test):
            with open("/workspaces/src/aruco_detector/config/aruco_location.yaml", 'r') as f:
                full_config = yaml.safe_load(f)

            self.unflipped_ids = full_config.get('unflipped_ids', [])
            self.marker_config = {int(k): v for k, v in full_config.items() if k != 'unflipped_ids'}
        else:
            self.config_sub = self.create_subscription(
                ArucoMarkerConfig,
                '/aruco_marker/config',
                self.config_callback,
                10
            )
            self.unflipped_ids = []
            self.marker_config = {}

        self.image_queue = deque(maxlen=5)  
        self.create_timer(0.05, self.process_image_queue) 

        threading.Thread(target=self.key_listener, daemon=True).start()

    def update_odometry_timer(self):
        now = self.get_clock().now().nanoseconds / 1e9
        dt = now - self.last_time
        self.last_time = now

        if dt <= 0.0:
            return

        self.odom_pose = self.update_odometry_from_four_wheels(
            self.front_data, self.rear_data, self.odom_pose, dt, 1.15
        )

        # x, y, theta = self.odom_pose
        # self.get_logger().info(f"Odometry: x={x:.2f}, y={y:.2f}, θ={np.degrees(theta):.1f}°")

    def rear_callback(self, msg):
        self.rear_data = msg.data

    def front_callback(self, msg):
        self.front_data = msg.data

    def update_odometry_from_four_wheels(self,
        front_wheel_data: list,
        rear_wheel_data: list,
        last_pose: np.ndarray,
        dt: float,
        wheel_radius: float = 0.04,
        wheel_base: float = 0.23,
        scale:float = 1.0
    ) -> np.ndarray:
        def compute_pose(wheel_l, wheel_r, pose):
            omega_l = wheel_l *scale
            omega_r = wheel_r *scale
            v_l = wheel_radius * omega_l
            v_r = wheel_radius * omega_r
            v = (v_r + v_l) / 2.0
            omega = (v_r - v_l) / wheel_base

            x, y, theta = pose
            x += v * np.cos(theta) * dt
            y += v * np.sin(theta) * dt
            theta += omega * dt
            return np.array([x, y, theta])
        
        pose_front = compute_pose(front_wheel_data[0], front_wheel_data[1], last_pose)
        pose_rear = compute_pose(rear_wheel_data[0], rear_wheel_data[1], last_pose)

        pose_avg = (pose_front + pose_rear) / 2.0
        pose_avg[2] = np.arctan2(np.sin(pose_avg[2]), np.cos(pose_avg[2]))  # 角度正規化 [-pi, pi]
        return pose_avg
        # omega_l = (front_wheel_data[0] + rear_wheel_data[0]) / 2.0
        # omega_r = (front_wheel_data[1] + rear_wheel_data[1]) / 2.0

        # v_l = wheel_radius * omega_l
        # v_r = wheel_radius * omega_r

        # v = (v_r + v_l) / 2.0
        # omega = (v_r - v_l) / wheel_base

        # x, y, theta = last_pose
        # x += v * np.cos(theta) * dt
        # y += v * np.sin(theta) * dt
        # theta += omega * dt

        # return np.array([x, y, theta])
    
    def key_listener(self):
        import sys
        import termios
        import tty
        import select
        import rclpy

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while rclpy.ok():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    if key.lower() == 'q':
                        rclpy.shutdown()
                        break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

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
        if self.marker_config:
            self.image_queue.append(msg)

    def process_image_queue(self):
        if not self.image_queue:
            return
   
        # Dequeue image
        msg = self.image_queue.popleft()
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")
        if cv_image is None:
            return
        
        # Estimate pose
        pose_list = self.process_markers(cv_image)
        
        if not pose_list:
            x, y, theta = self.odom_pose
            odom_quat = np.array([
                0.0, 0.0,
                np.sin(theta / 2.0),
                np.cos(theta / 2.0)
            ])

            pose_msg = PoseWithCovarianceStamped()
            pose_msg.header.frame_id = "map"
            pose_msg.pose.pose.position.x = x
            pose_msg.pose.pose.position.y = y
            pose_msg.pose.pose.position.z = 0.0
            pose_msg.pose.pose.orientation.x = odom_quat[0]
            pose_msg.pose.pose.orientation.y = odom_quat[1]
            pose_msg.pose.pose.orientation.z = odom_quat[2]
            pose_msg.pose.pose.orientation.w = odom_quat[3]
            self.pose_pub.publish(pose_msg)
            return
        
        # Marker-based estimation
        pos_arr = np.array([pose[:3] for pose in pose_list])
        q_arr = np.array([pose[3:] for pose in pose_list])

        if hasattr(self, "last_pos"):
            dists = np.linalg.norm(pos_arr - self.last_pos, axis=1)
            filtered_idx = np.where(dists < 0.8)[0]  # filter outliers
            if len(filtered_idx) == 0:
                return # TODO aslo publish odom pose
            pos_arr = pos_arr[filtered_idx]
            q_arr = q_arr[filtered_idx]

        avg_pos = np.mean(pos_arr, axis=0)
        avg_quat = np.mean(q_arr, axis=0)
        avg_quat /= np.linalg.norm(avg_quat)

        # Fusion with odometry
        self.last_pos, self.last_quat = self.fuse_pose_with_odom(avg_pos, avg_quat, self.odom_pose)

        # Publish final pose
        print(f"{self.last_pos}")
        pose_msg = PoseWithCovarianceStamped()
        pose_msg.header.frame_id = "map"
        pose_msg.pose.pose.position.x = self.last_pos[0]
        pose_msg.pose.pose.position.y = self.last_pos[1]
        pose_msg.pose.pose.position.z = self.last_pos[2]
        pose_msg.pose.pose.orientation.x = self.last_quat[0]
        pose_msg.pose.pose.orientation.y = self.last_quat[1]
        pose_msg.pose.pose.orientation.z = self.last_quat[2]
        pose_msg.pose.pose.orientation.w = self.last_quat[3]
        self.pose_pub.publish(pose_msg)

        # === DEBUG PUBLISH: Odometry-only Pose ===
        x, y, theta = self.odom_pose
        odom_quat = np.array([
            0.0, 0.0,
            np.sin(theta / 2.0),
            np.cos(theta / 2.0)
        ])
        odom_pose_msg = PoseWithCovarianceStamped()
        odom_pose_msg.header.frame_id = "map"
        odom_pose_msg.pose.pose.position.x = x
        odom_pose_msg.pose.pose.position.y = y
        odom_pose_msg.pose.pose.position.z = 0.0
        odom_pose_msg.pose.pose.orientation.x = odom_quat[0]
        odom_pose_msg.pose.pose.orientation.y = odom_quat[1]
        odom_pose_msg.pose.pose.orientation.z = odom_quat[2]
        odom_pose_msg.pose.pose.orientation.w = odom_quat[3]
        self.debug_odom_pub.publish(odom_pose_msg)

        # === DEBUG PUBLISH: ArUco marker avg Pose ===
        avg_pos_msg = PoseWithCovarianceStamped()
        avg_pos_msg.header.frame_id = "map"
        avg_pos_msg.pose.pose.position.x = avg_pos[0]
        avg_pos_msg.pose.pose.position.y = avg_pos[1]
        avg_pos_msg.pose.pose.position.z = avg_pos[2]
        avg_pos_msg.pose.pose.orientation.x = avg_quat[0]
        avg_pos_msg.pose.pose.orientation.y = avg_quat[1]
        avg_pos_msg.pose.pose.orientation.z = avg_quat[2]
        avg_pos_msg.pose.pose.orientation.w = avg_quat[3]
        self.debug_marker_pub.publish(avg_pos_msg)

        
        # >> For Debug
        if self.debug:
            debug_img = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            self.image_pub.publish(debug_img)

    def fuse_pose_with_odom(self, avg_pos, avg_quat, odom_pose, w_marker=0.7, w_odom=0.3):
        odom_pos = np.array([odom_pose[0], odom_pose[1], 0.0])
        fused_pos = w_marker * avg_pos + w_odom * odom_pos

        # >> odom to quat
        odom_yaw = odom_pose[2]
        odom_quat = np.array([
            0.0, 0.0,
            np.sin(odom_yaw / 2.0),
            np.cos(odom_yaw / 2.0)
        ])

        fused_quat = w_marker * avg_quat + w_odom * odom_quat
        fused_quat /= np.linalg.norm(fused_quat)

        return fused_pos, fused_quat
    
    def process_markers(self, image):
        result_poses = []
        # >> Pre-process
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        # >> Detect markers
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.config.aruco_dict, parameters=self.config.aruco_params)
        if ids is None:
            return []
        
        # >> Refine corners
        gray_float = np.float32(gray)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        for i in range(len(corners)):
            cv2.cornerSubPix(gray_float, corners[i], winSize=(5, 5), zeroZone=(-1, -1), criteria=criteria)

        # >> For Debug
        if self.debug : cv2.aruco.drawDetectedMarkers(image, corners, ids)

        # >> Process each marker
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id not in self.marker_config or self.polygon_area(corners[i]) < 1500:
                continue

            # >> Reorder marker corners
            image_points = corners[i]
            if marker_id not in self.unflipped_ids:
                reorder = [2, 3, 0, 1]
                image_points = image_points[:, reorder, :]

            # >> Estimate marker pose in camera frame
            success, rvec, tvec, _ = cv2.solvePnPRansac(
                self.config.objp, image_points,
                self.config.camera_matrix, self.config.dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
                reprojectionError=3.0, confidence=0.99, iterationsCount=100
            )
            if not success:
                continue
            
            if self.debug : 
                cv2.drawFrameAxes(image, self.config.camera_matrix, self.config.dist_coeffs, rvec, tvec, self.config.marker_length * 0.5)
            
            # >> Transform from camera frame to map frame
            R, _ = cv2.Rodrigues(rvec)
            T_camera_marker = np.eye(4)
            T_camera_marker[:3, :3] = R
            T_camera_marker[:3, 3] = tvec.flatten()

            m = self.marker_config[marker_id]
            theta = m['theta']
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            R_map_marker = np.array([
                [cos_t, -sin_t, 0], 
                [sin_t, cos_t,  0], 
                [0,     0,      1]])
            T_map_marker = np.eye(4)
            T_map_marker[:3, :3] = R_map_marker
            T_map_marker[:3, 3] = np.array([m['x'], m['y'], 0])

            T_map_camera = T_map_marker @ self.config.T_align @ np.linalg.inv(T_camera_marker)
            T_map_base = T_map_camera @ np.linalg.inv(self.config.T_camera_base)

            pos = T_map_base[:3, 3]
            R_base = T_map_base[:3, :3]
            quat = scipy_R.from_matrix(R_base).as_quat()
            result_poses.append(np.concatenate([pos, quat]))

        return result_poses

    def yaw_to_quaternion(self, yaw):
        qx = 0.0
        qy = 0.0
        qz = np.sin(yaw / 2.0)
        qw = np.cos(yaw / 2.0)
        return (qx, qy, qz, qw)
        
    def scene_reset_callback(self, msg: Bool):
        if msg.data: 
            self.get_logger().info("Scene change requested. Resetting state...")
            self.reset_internal_state()

    def reset_internal_state(self):
        self.last_pos = None
        self.last_quat = None
        self.image_queue.clear()
        self.odom_pose = [0.0, 0.0, 0.0]
        self.get_logger().info("Internal state has been reset.")

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
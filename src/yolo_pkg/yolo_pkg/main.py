import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image, Imu
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import numpy as np
import yaml
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory
import os
import contextlib
import io
from scipy.spatial.transform import Rotation as R
from geometry_msgs.msg import PointStamped


class YoloDetectionNode(Node):
    def __init__(self):
        super().__init__('yolo_detection_node')
        
        # 初始化 cv_bridge
        self.bridge = CvBridge()
        
        # 初始化 YOLO 模型，使用 GPU
        model_path = os.path.join(
            get_package_share_directory('yolo_pkg'),
            'models',
            'best_nano_auto_augu_super_close_fire.pt'
        )
        self.model = YOLO(model_path)
        self.model.to('cuda')
        
        # 載入相機內參
        self.camera_intrinsics = self.load_camera_intrinsics()

        # 訂閱 RGB 和深度訊息
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/camera/color/image_raw/compressed',
            self.image_callback,
            10
        )
        
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10
        )

        # 訂閱 /odometry/filtered 和 /imu/data_raw 訊息
        self.odom_sub = self.create_subscription(Odometry, '/odometry/filtered', self.odom_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/imu/data_raw', self.imu_callback, 10)

        # 建立影像發布者，發布偵測後的影像
        self.image_pub = self.create_publisher(CompressedImage, '/yolo/detection/compressed', 10)

        # 建立物體位置的發布者
        self.point_pub = self.create_publisher(PointStamped, '/yolo/detection/position', 10)
        
        # 初始化深度影像存儲
        self.depth_image = None

        # 儲存平移和旋轉資訊
        self.translation = None
        self.rotation_matrix = None

    def load_camera_intrinsics(self):
        """載入相機內參"""
        config_path = os.path.join(get_package_share_directory('yolo_pkg'), 'config', 'ost.yaml')
        with open(config_path, 'r') as file:
            camera_params = yaml.safe_load(file)
        
        return {
            "fx": camera_params['camera_matrix']['data'][0],
            "fy": camera_params['camera_matrix']['data'][4],
            "cx": camera_params['camera_matrix']['data'][2],
            "cy": camera_params['camera_matrix']['data'][5]
        }

    def odom_callback(self, msg):
        """處理 /odometry/filtered 訊息以獲取平移向量"""
        self.translation = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])

    def imu_callback(self, msg):
        """處理 /imu/data_raw 訊息以獲取旋轉矩陣"""
        orientation_q = msg.orientation
        r = R.from_quat([orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w])
        self.rotation_matrix = r.as_matrix()

    def publish_position(self, position):
        """發布物體的全域座標"""
        point_msg = PointStamped()
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.header.frame_id = "map"  # 可以根據需求設置參考座標系
        point_msg.point.x = position[0]
        point_msg.point.y = position[1]
        point_msg.point.z = position[2]
        
        self.point_pub.publish(point_msg)
        self.get_logger().info(f"Published object global position: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]")


    def image_callback(self, msg):
        """RGB影像處理與目標檢測"""
        # 將影像消息轉換為 OpenCV 格式
        cv_image = self.convert_image(msg)
        
        if cv_image is None:
            self.get_logger().error("Failed to convert image.")
            return
        
        # 進行 YOLO 物體檢測
        detection_results = self.detect_objects(cv_image)
        
        # 繪製 Bounding Box 並計算深度
        processed_image = self.draw_bounding_boxes(cv_image, detection_results)
        
        # 將處理後的影像發布
        self.publish_image(processed_image)

    def depth_callback(self, msg):
        """更新最新的深度影像"""
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def convert_image(self, msg):
        """將 ROS 影像消息轉換為 OpenCV 格式"""
        try:
            return self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return None

    def detect_objects(self, image):
        """使用 YOLO 模型進行物體檢測，並抑制輸出訊息"""
        with contextlib.redirect_stdout(io.StringIO()):
            results = self.model(image, verbose=False)
        return results

    def draw_bounding_boxes(self, image, results):
        """在影像上繪製 YOLO 偵測到的 Bounding Box 並取得物體深度及在全域座標系中的3D座標"""
        if self.depth_image is None:
            self.get_logger().warning("No depth image available.")
            return image

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 計算物體中心點的深度
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                depth_value = self.get_depth(center_x, center_y)

                # 計算在相機座標系中的3D座標
                if depth_value > 0:  # 深度值為正數才計算
                    object_position_camera = self.calculate_3d_position(center_x, center_y, depth_value)
                    
                    # 使用外參將座標從相機座標系轉換到全域座標系
                    object_position_global = self.transform_to_global(object_position_camera)
                    self.publish_position(object_position_global)
                    label1 = f"Conf: {float(box.conf):.2f}, Depth: {depth_value:.2f} m"
                    label2 = f"Global Pos: [{object_position_global[0]:.2f}, {object_position_global[1]:.2f}, {object_position_global[2]:.2f}]"
                else:
                    label1 = f"Conf: {float(box.conf):.2f}, Depth: N/A"
                    label2 = "Global Pos: N/A"

                # 在影像上繪製兩行文字
                cv2.putText(image, label1, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(image, label2, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        return image


    def calculate_3d_position(self, x, y, depth):
        """使用內參和深度計算物體的3D座標"""
        fx = self.camera_intrinsics['fx']
        fy = self.camera_intrinsics['fy']
        cx = self.camera_intrinsics['cx']
        cy = self.camera_intrinsics['cy']

        # 計算3D空間中的座標（假設深度單位為米）
        X = (x - cx) * depth / fx
        Y = (y - cy) * depth / fy
        Z = depth

        return np.array([X, Y, Z])

    def get_depth(self, x, y):
        """取得指定像素點的深度，以公尺為單位"""
        if self.depth_image is not None:
            depth_value = self.depth_image[y, x]
            # 深度影像單位可能是毫米，轉換為公尺
            return depth_value / 1000.0
        else:
            return 0.0
    
    def transform_to_global(self, position_camera):
        """将物体位置从相机坐标系转换到全局坐标系"""
        if self.translation is None or self.rotation_matrix is None:
            self.get_logger().warning("Translation or rotation matrix not available.")
            return position_camera  # 若无外参则返回相机坐标

        # 定义IMU到相机坐标系的旋转矩阵
        R_imu_cam = np.array([
            [1,  0,  0],
            [0, -1,  0],
            [0,  0, -1]
        ])

        # 调整后的旋转矩阵
        R_world_cam = self.rotation_matrix @ R_imu_cam

        # 验证旋转矩阵
        if not np.allclose(R_world_cam @ R_world_cam.T, np.eye(3), atol=1e-6):
            self.get_logger().warning("Rotation matrix is not orthogonal.")
        if not np.isclose(np.linalg.det(R_world_cam), 1.0, atol=1e-6):
            self.get_logger().warning("Determinant of rotation matrix is not 1.")

        # 创建4x4的转换矩阵（外参矩阵）
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = R_world_cam
        extrinsics[:3, 3] = self.translation

        # 将相机坐标扩展为齐次坐标
        position_camera_homogeneous = np.append(position_camera, 1)

        # 转换到全局坐标
        position_global_homogeneous = extrinsics.dot(position_camera_homogeneous)

        # 提取全局坐标中的 x, y, z
        return position_global_homogeneous[:3]



    def publish_image(self, image):
        """將處理後的影像轉換並發布到 ROS"""
        try:
            compressed_msg = self.bridge.cv2_to_compressed_imgmsg(image)
            self.image_pub.publish(compressed_msg)
        except Exception as e:
            self.get_logger().error(f"Could not publish image: {e}")

    def get_extrinsics(self):
        """取得相機外部參數"""
        if self.translation is None or self.rotation_matrix is None:
            self.get_logger().warning("Translation or rotation matrix not available.")
            return None
        # 將平移向量和旋轉矩陣組成外參
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = self.rotation_matrix
        extrinsics[:3, 3] = self.translation
        return extrinsics


def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

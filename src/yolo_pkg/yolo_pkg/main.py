import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image, Imu
from cv_bridge import CvBridge
import cv2
import numpy as np
import yaml
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory
import os
import contextlib
import io
from geometry_msgs.msg import PointStamped
from scipy.spatial.transform import Rotation as R
from std_msgs.msg import String, Bool

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
        
        # 载入相机内参
        self.camera_intrinsics = self.load_camera_intrinsics()

        # 订阅 RGB 和深度信息
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/camera/color/image_raw/compressed',
            self.image_callback,
            10
        )
        # 订阅 IMU 数据
        self.imu_orientation = None
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',  # 替换为您的 IMU 话题名称
            self.imu_callback,
            10
        )
        
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10
        )

        self.target_label_sub = self.create_subscription(
            String,
            '/target_label',
            self.target_label_callback,
            10
        )

        # 创建影像发布者，发布检测后的影像
        self.image_pub = self.create_publisher(CompressedImage, '/yolo/detection/compressed', 10)

        # 创建物体位置的发布者
        self.point_pub = self.create_publisher(PointStamped, '/yolo/detection/position', 10)

        self.point_offset_pub = self.create_publisher(PointStamped, '/yolo/detection/offset', 10)
        self.detection_status_pub = self.create_publisher(Bool, '/yolo/detection/status', 10)
        
        # 初始化深度影像存储
        self.depth_image = None
        self.target_label = None
    
    def imu_callback(self, msg):
        """接收并存储 IMU 的姿态信息"""
        self.imu_orientation = msg.orientation
    
    def target_label_callback(self, msg):
        """Callback to receive the target label name"""
        self.target_label = msg.data

    def get_imu_rotation_matrix(self):
        """获取 IMU 的旋转矩阵"""
        if self.imu_orientation is None:
            self.get_logger().warning("No IMU orientation data available.")
            return np.eye(3)  # 返回单位矩阵，表示没有旋转

        # 从四元数构造旋转矩阵
        quat = [
            self.imu_orientation.x,
            self.imu_orientation.y,
            self.imu_orientation.z,
            self.imu_orientation.w
        ]
        r = R.from_quat(quat)
        rotation_matrix = r.as_matrix()
        return rotation_matrix
    
    def calculate_movement_to_center_crosshair(self, object_center_x, object_center_y, depth_value):
        """
        計算需要的 3D 位移，使機械手臂的末端移動，讓畫面中心十字架對準物體中心點。
        Args:
            object_center_x (int): 物體在畫面中的 x 座標
            object_center_y (int): 物體在畫面中的 y 座標
            depth_value (float): 物體的深度
        Returns:
            np.array: 3D 位移向量，用於使十字架對準物體
        """
        # 相機內參
        fx = self.camera_intrinsics['fx']
        fy = self.camera_intrinsics['fy']
        cx = self.camera_intrinsics['cx']
        cy = self.camera_intrinsics['cy']
        
        # 計算物體相對於畫面中心（十字架）的偏移量
        x_offset = (object_center_x - cx) * depth_value / fx
        y_offset = (object_center_y - cy) * depth_value / fy
        z_offset = 0  # 保持原深度

        # 3D 位移向量，不使用 IMU，僅根據相機資訊進行偏移
        offset_in_camera_frame = np.array([x_offset, -y_offset, z_offset])
        
        return offset_in_camera_frame


    def load_camera_intrinsics(self):
        """载入相机内参"""
        config_path = os.path.join(get_package_share_directory('yolo_pkg'), 'config', 'ost.yaml')
        with open(config_path, 'r') as file:
            camera_params = yaml.safe_load(file)
        
        return {
            "fx": camera_params['camera_matrix']['data'][0],
            "fy": camera_params['camera_matrix']['data'][4],
            "cx": camera_params['camera_matrix']['data'][2],
            "cy": camera_params['camera_matrix']['data'][5]
        }

    def publish_position(self, position):
        """发布物体在相机坐标系下的坐标"""
        point_msg = PointStamped()
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.header.frame_id = "camera_link"  # 设置为相机坐标系
        point_msg.point.x = position[0]
        point_msg.point.y = position[1]
        point_msg.point.z = position[2]
        
        self.point_pub.publish(point_msg)
        # self.get_logger().info(f"Published object position in camera frame: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]")
    
    def publish_offset(self, position):
        point_msg = PointStamped()
        point_msg.header.stamp = self.get_clock().now().to_msg()
        point_msg.point.x = position[0]
        point_msg.point.y = position[1]
        point_msg.point.z = position[2]
        
        self.point_offset_pub.publish(point_msg)
        # self.get_logger().info(f"Published object position in camera frame: [{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}]")

    def image_callback(self, msg):
        """RGB影像处理与目标检测"""
        # 将影像消息转换为 OpenCV 格式
        cv_image = self.convert_image(msg)
        
        if cv_image is None:
            self.get_logger().error("Failed to convert image.")
            return
        
        # 进行 YOLO 物体检测
        detection_results = self.detect_objects(cv_image)
        
        # 绘制 Bounding Box 并计算深度
        processed_image = self.draw_bounding_boxes(cv_image, detection_results)
        
        # 将处理后的影像发布
        self.publish_image(processed_image)

    def depth_callback(self, msg):
        """更新最新的深度影像"""
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')

    def convert_image(self, msg):
        """将 ROS 影像消息转换为 OpenCV 格式"""
        try:
            return self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return None

    def detect_objects(self, image):
        """使用 YOLO 模型进行物体检测，并抑制输出信息"""
        with contextlib.redirect_stdout(io.StringIO()):
            results = self.model(image, verbose=False)
        return results

    def draw_bounding_boxes(self, image, results):
        """在影像上绘制 YOLO 检测到的 Bounding Box 并获取物体深度及在相机坐标系中的3D坐标"""
        if self.depth_image is None:
            self.get_logger().warning("No depth image available.")
            return image
        
        cx = int(self.camera_intrinsics['cx'])
        cy = int(self.camera_intrinsics['cy'])
        crosshair_color = (0, 0, 255)  # 红色
        crosshair_thickness = 2
        crosshair_length = 20  # 十字架线段长度
        cv2.line(
            image,
            (cx - crosshair_length, cy),
            (cx + crosshair_length, cy),
            crosshair_color,
            crosshair_thickness
        )
        cv2.line(
            image,
            (cx, cy - crosshair_length),
            (cx, cy + crosshair_length),
            crosshair_color,
            crosshair_thickness
        )
        target_detected = False
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                if class_name != self.target_label:
                    continue
                target_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 计算物体中心点的深度
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)
                depth_value = self.get_depth(center_x, center_y)
                movement_offset = self.calculate_movement_to_center_crosshair(center_x, center_y, depth_value)
                
                # 计算在相机坐标系中的3D坐标
                if depth_value > 0:  # 深度值为正数才计算
                    object_position_camera = self.calculate_3d_position(center_x, center_y, depth_value)
                    
                    # object_position_camera = self.correct_position_with_imu(object_position_camera)
                    # object_position_camera = self.transform_to_left_hand_coordinate(object_position_camera)
                    self.publish_offset(movement_offset)
                    self.publish_position(object_position_camera)
                    label1 = f"Conf: {float(box.conf):.2f}, Depth: {depth_value:.2f} m"
                    label2 = f"Pos: [{object_position_camera[0]:.2f}, {object_position_camera[1]:.2f}, {object_position_camera[2]:.2f}]"
                else:
                    label1 = f"Conf: {float(box.conf):.2f}, Depth: N/A"
                    label2 = "Pos: N/A"

                # 在影像上绘制两行文字
                cv2.putText(image, label1, (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(image, label2, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        self.detection_status_pub.publish(Bool(data=target_detected))  
        return image

    def calculate_3d_position(self, x, y, depth):
        fx = self.camera_intrinsics['fx']
        fy = self.camera_intrinsics['fy']
        cx = self.camera_intrinsics['cx']
        cy = self.camera_intrinsics['cy']

        X = (x - cx) * depth / fx  # X 轴：水平，向右为正
        Y = (y - cy) * depth / fy  # Y 轴：垂直，向下为正
        Z = depth                  # Z 轴：指向前方，深度方向

        # 將相機坐標系轉換為 IMU 坐標系
        imu_position = np.array([Z, -X, -Y])

        return imu_position
    
    def calculate_3d_position_imu(self, x, y, depth):
        # 使用相機內參將像素坐標轉換為相機坐標系下的 3D 座標
        fx = self.camera_intrinsics['fx']
        fy = self.camera_intrinsics['fy']
        cx = self.camera_intrinsics['cx']
        cy = self.camera_intrinsics['cy']

        X = (x - cx) * depth / fx
        Y = (y - cy) * depth / fy
        Z = depth

        # 相機坐標系下的物體位置
        object_position_camera = np.array([X, Y, Z])

        # 獲取 IMU 的旋轉矩陣，並應用到相機坐標系下的物體位置
        imu_rotation_matrix = self.get_imu_rotation_matrix()
        object_position_world = imu_rotation_matrix @ object_position_camera

        return object_position_world

    
    def transform_to_left_hand_coordinate(self, position):
        # 只需反轉 Y 軸即可
        new_position = np.array([position[0], -position[1], position[2]])
        return new_position


    def get_depth(self, x, y):
        """获取指定像素点的深度，以米为单位"""
        if self.depth_image is not None:
            try:
                depth_value = self.depth_image[y, x]
                # 深度影像单位可能是毫米，转换为米
                return depth_value / 1000.0
            except IndexError:
                self.get_logger().warning("Depth image index out of bounds.")
                return 0.0
        else:
            print("depth_image is None")
            return 0.0

    def publish_image(self, image):
        """将处理后的影像转换并发布到 ROS"""
        try:
            compressed_msg = self.bridge.cv2_to_compressed_imgmsg(image)
            self.image_pub.publish(compressed_msg)
        except Exception as e:
            self.get_logger().error(f"Could not publish image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = YoloDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

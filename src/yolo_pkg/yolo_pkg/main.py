import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import yaml
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory
import os

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

        # 建立影像發布者，發布偵測後的影像
        self.image_pub = self.create_publisher(CompressedImage, '/yolo/detection/compressed', 10)
        
        # 初始化深度影像存儲
        self.depth_image = None

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
        """使用 YOLO 模型進行物體檢測"""
        return self.model(image)

    def draw_bounding_boxes(self, image, results):
        """在影像上繪製 YOLO 偵測到的 Bounding Box 並取得物體深度"""
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

                label = f"{int(box.cls)}: {float(box.conf):.2f}, Depth: {depth_value:.2f} m"
                cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
        return image

    def get_depth(self, x, y):
        """取得指定像素點的深度，以公尺為單位"""
        if self.depth_image is not None:
            depth_value = self.depth_image[y, x]
            # 深度影像單位可能是毫米，轉換為公尺
            return depth_value / 1000.0
        else:
            return 0.0

    def publish_image(self, image):
        """將處理後的影像轉換並發布到 ROS"""
        try:
            compressed_msg = self.bridge.cv2_to_compressed_imgmsg(image)
            self.image_pub.publish(compressed_msg)
            # self.get_logger().info("Published processed image with bounding boxes.")
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

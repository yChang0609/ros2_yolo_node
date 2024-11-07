import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import torch
from ultralytics import YOLO
import numpy as np
from ament_index_python.packages import get_package_share_directory
import os
class YoloDetectionNode(Node):
    def __init__(self):
        super().__init__('yolo_detection_node')
        
        # 初始化 cv_bridge
        self.bridge = CvBridge()
        
        # 初始化 YOLO 模型，使用 GPU
        # 獲取模型文件的路徑，並初始化 YOLO 模型
        model_path = os.path.join(
            get_package_share_directory('yolo_pkg'),
            'models',
            'best_nano_auto_augu_super_close_fire.pt'
        )
        self.model = YOLO(model_path)
        self.model.to('cuda')
        
        # 訂閱攝像頭的影像訊息
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/camera/color/image_raw/compressed',
            self.image_callback,
            10
        )
        
        # 建立影像發布者，發布偵測後的影像
        self.image_pub = self.create_publisher(CompressedImage, '/yolo/detection/compressed', 10)

    def image_callback(self, msg):
        # 使用 cv_bridge 將影像消息轉換為 OpenCV 格式
        print("image_callback")
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return

        # 使用 YOLO 進行物體檢測
        results = self.model(cv_image)

        # 繪製 Bounding Box
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{int(box.cls)}: {float(box.conf):.2f}"
                cv2.putText(cv_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 將偵測後的影像轉換回 ROS 消息並發布
        try:
            compressed_msg = self.bridge.cv2_to_compressed_imgmsg(cv_image)
            self.image_pub.publish(compressed_msg)
            self.get_logger().info("Published processed image with bounding boxes.")
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

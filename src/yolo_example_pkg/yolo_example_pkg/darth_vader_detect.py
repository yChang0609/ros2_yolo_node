import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import os
from ament_index_python.packages import get_package_share_directory
import torch


class YoloDetectionNode(Node):
    def __init__(self):
        super().__init__("yolo_detection_node")

        # 初始化 cv_bridge
        self.bridge = CvBridge()

        self.latest_depth_image_raw = None
        self.latest_depth_image_compressed = None

        # 使用 yolo model 位置
        model_path = os.path.join(
            get_package_share_directory("yolo_example_pkg"), "models", "darth_vader.pt"
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print("Using device : ", device)
        self.model = YOLO(model_path)
        self.model.to(device)

        # 訂閱影像 Topic
        self.image_sub = self.create_subscription(
            CompressedImage, "/camera/image/compressed", self.image_callback, 10
        )

        # 訂閱 **無壓縮** 深度圖 Topic
        self.depth_sub_raw = self.create_subscription(
            Image, "/camera/depth/image_raw", self.depth_callback_raw, 10
        )

        # 訂閱 **壓縮** 深度圖 Topic
        self.depth_sub_compressed = self.create_subscription(
            CompressedImage,
            "/camera/depth/compressed",
            self.depth_callback_compressed,
            10,
        )

        # 發佈處理後的影像 Topic
        self.image_pub = self.create_publisher(
            CompressedImage, "/yolo/detection/compressed", 10
        )

        # 設定要過濾標籤 (如果為空，那就不過濾)
        self.allowed_labels = {}

        # 設定 YOLO 可信度閾值
        self.conf_threshold = 0.5  # 可以修改這個值來調整可信度

    def depth_callback_raw(self, msg):
        """接收 **無壓縮** 深度圖"""
        try:
            self.latest_depth_image_raw = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough"
            )
        except Exception as e:
            self.get_logger().error(f"Could not convert raw depth image: {e}")

    def depth_callback_compressed(self, msg):
        """接收 **壓縮** 深度圖（當無壓縮深度圖不可用時使用）"""
        try:
            self.latest_depth_image_compressed = self.bridge.compressed_imgmsg_to_cv2(
                msg, desired_encoding="passthrough"
            )
        except Exception as e:
            self.get_logger().error(f"Could not convert compressed depth image: {e}")

    def image_callback(self, msg):
        """接收影像並進行物體檢測"""
        # 將 ROS 影像消息轉換為 OpenCV 格式
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(
                msg, desired_encoding="bgr8"
            )
        except Exception as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return

        # 使用 YOLO 模型檢測物體
        try:
            results = self.model(cv_image, conf=self.conf_threshold, verbose=False)
        except Exception as e:
            self.get_logger().error(f"Error during YOLO detection: {e}")
            return

        # 繪製 Bounding Box
        processed_image = self.draw_bounding_boxes(cv_image, results)

        # 發佈處理後的影像
        self.publish_image(processed_image)

    def draw_bounding_boxes(self, image, results):
        """在影像上繪製 YOLO 檢測到的 Bounding Box"""
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]

                # 只保留設定內的標籤
                if self.allowed_labels and class_name not in self.allowed_labels:
                    continue
                # 計算 Bounding Box 正中心點
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                # 優先使用無壓縮的深度圖
                depth_value = self.get_depth_at(cx, cy)
                depth_text = f"{depth_value:.2f}m" if depth_value else "N/A"

                # 繪製框和標籤
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {conf:.2f} Depth: {depth_text}"
                cv2.putText(
                    image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        return image

    def get_depth_at(self, x, y):
        """取得指定像素的深度值，轉換為米 (m)"""
        # **優先使用無壓縮的深度圖**
        depth_image = (
            self.latest_depth_image_raw
            if self.latest_depth_image_raw is not None
            else self.latest_depth_image_compressed
        )

        if depth_image is None:
            return None

        # 如果深度影像為三通道，那只取第一個數值
        if len(depth_image.shape) == 3:
            depth_image = depth_image[:, :, 0]

        try:
            depth_value = depth_image[y, x]
            if depth_value == 0:  # 無效深度
                return None
            return depth_value / 1000.0  # 16-bit 深度圖通常單位為 mm，轉換為 m
        except IndexError:
            return None

    def publish_image(self, image):
        """將處理後的影像轉換並發佈到 ROS"""
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


if __name__ == "__main__":
    main()

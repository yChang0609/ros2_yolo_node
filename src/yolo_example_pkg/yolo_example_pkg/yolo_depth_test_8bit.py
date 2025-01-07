import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import os
from ament_index_python.packages import get_package_share_directory


class YoloDepthDetectionNode(Node):
    def __init__(self):
        super().__init__("yolo_depth_detection_node")

        # 初始化 cv_bridge
        self.bridge = CvBridge()

        # 加載 YOLO 模型
        model_path = os.path.join(
            get_package_share_directory("yolo_pkg"), "models", "yolov8n.pt"
        )
        self.model = YOLO(model_path)
        self.model.to("cuda")

        # 訂閱 RGB 影像 Topic
        self.image_sub = self.create_subscription(
            CompressedImage, "/camera/image/compressed", self.image_callback, 10
        )

        # 訂閱壓縮深度影像 Topic
        self.depth_sub = self.create_subscription(
            CompressedImage, "/camera/depth/compressed", self.depth_callback, 10
        )

        # 發佈處理後的影像 Topic
        self.image_pub = self.create_publisher(
            CompressedImage, "/yolo/detection/compressed", 10
        )

        # 深度影像緩存
        self.depth_image = None

    def depth_callback(self, msg):
        """接收壓縮深度影像並解碼"""
        try:
            # 解碼 Base64 數據為 PNG 字節數組
            png_data = np.frombuffer(msg.data, dtype=np.uint8)

            # 解壓 PNG 到單通道影像
            depth_image = cv2.imdecode(png_data, cv2.IMREAD_UNCHANGED)

            if depth_image is None:
                self.get_logger().error("Decoded depth image is None.")
                return

            self.depth_image = depth_image
            self.get_logger().info(
                f"Received depth image. Shape: {self.depth_image.shape}"
            )
        except Exception as e:
            self.get_logger().error(f"Could not decode depth image: {e}")

    def image_callback(self, msg):
        """接收影像並進行物體檢測"""
        try:
            cv_image = self.bridge.compressed_imgmsg_to_cv2(
                msg, desired_encoding="bgr8"
            )
        except Exception as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return

        if self.depth_image is None:
            self.get_logger().warn("No depth image available yet.")
            return

        try:
            results = self.model(cv_image, verbose=False)
        except Exception as e:
            self.get_logger().error(f"Error during YOLO detection: {e}")
            return

        processed_image = self.draw_bounding_boxes(cv_image, results)
        self.publish_image(processed_image)

    def draw_bounding_boxes(self, image, results):
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                depth = self.get_depth_at_point(cx, cy)

                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {conf:.2f} Depth: {depth:.2f} meters"
                cv2.putText(
                    image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                self.get_logger().info(
                    f"Detected {class_name} at depth {depth:.2f} meters."
                )

        return image

    def get_depth_at_point(self, x, y):
        """從深度影像中獲取指定點的深度值"""
        try:
            depth_value = self.depth_image[y, x]
            # 假設深度範圍 [0, 65535]，轉換到米
            max_depth_meters = 10.0  # 根據 Unity 發佈的深度範圍設定
            depth_in_meters = (depth_value / 255.0) * max_depth_meters
            return depth_in_meters
        except IndexError:
            self.get_logger().warn(f"Invalid depth coordinates: ({x}, {y})")
            return float("nan")

    def publish_image(self, image):
        """發佈處理後的影像到 ROS"""
        try:
            compressed_msg = self.bridge.cv2_to_compressed_imgmsg(image)
            self.image_pub.publish(compressed_msg)
        except Exception as e:
            self.get_logger().error(f"Could not publish image: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = YoloDepthDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

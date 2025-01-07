import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
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
        self.depth_image_pub = self.create_publisher(
            CompressedImage, "/yolo/depth/detection/compressed", 10
        )

        # 訂閱深度影像 Topic
        self.depth_sub = self.create_subscription(
            Image, "/depth_image_topic", self.depth_callback, 10
        )

        # 發佈處理後的影像 Topic
        self.image_pub = self.create_publisher(
            CompressedImage, "/yolo/detection/compressed", 10
        )

        # 深度影像緩存
        self.depth_image = None

    def depth_callback(self, msg):
        """接收深度影像"""
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(
                msg, desired_encoding="passthrough"
            )
        except Exception as e:
            self.get_logger().error(f"Could not convert depth image: {e}")

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

        # 繪製檢測框到深度圖
        depth_with_boxes = self.draw_bounding_boxes_on_depth(self.depth_image, results)

        # 發佈帶檢測框的深度圖
        self.publish_depth_image(depth_with_boxes)

    def draw_bounding_boxes_on_depth(self, depth_image, results):
        """在深度圖上繪製 YOLO 檢測框"""
        # 將深度圖轉換為彩色圖像（伽馬校正 + 映射至顏色空間）
        depth_colored = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
        )

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]

                # 計算中心點
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # 獲取深度值
                depth = self.get_depth_at_point(cx, cy)

                # 繪製邊框和標籤
                cv2.rectangle(depth_colored, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {conf:.2f} Depth: {depth:.2f}m"
                cv2.putText(
                    depth_colored,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        return depth_colored

    def publish_depth_image(self, depth_image):
        """發佈帶有檢測框的深度圖"""
        try:
            depth_msg = self.bridge.cv2_to_compressed_imgmsg(depth_image)
            self.image_pub.publish(depth_msg)
        except Exception as e:
            self.get_logger().error(f"Could not publish depth image: {e}")

    def draw_bounding_boxes(self, image, results):
        """在影像上繪製 YOLO 檢測到的 Bounding Box 並顯示深度距離"""
        for result in results:
            for box in result.boxes:
                # 獲取檢測框座標
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf)
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]

                # 計算框中心點
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # 獲取深度值
                depth = self.get_depth_at_point(cx, cy)

                # 繪製框和標籤
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{class_name} {conf:.2f} Depth: {depth:.2f}m"
                cv2.putText(
                    image,
                    label,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

                # 輸出深度資訊
                self.get_logger().info(
                    f"Detected {class_name} at depth {depth:.2f} meters."
                )

        return image

    def get_depth_at_point(self, x, y):
        """從深度影像中獲取指定點的深度值"""
        try:
            depth = self.depth_image[y, x]  # 假設深度影像是單通道
            return depth / 1000.0  # 將深度值從毫米轉換為米
        except IndexError:
            self.get_logger().warn(f"Invalid depth coordinates: ({x}, {y})")
            return float("nan")

    def publish_image(self, image):
        """將處理後的影像轉換並發佈到 ROS"""
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

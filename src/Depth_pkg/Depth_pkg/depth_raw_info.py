#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np


class DepthImageInspector(Node):
    def __init__(self):
        super().__init__("depth_image_inspector")

        # 訂閱 depth_image_topic
        self.subscription = self.create_subscription(
            Image, "depth_image_topic", self.listener_callback, 10  # QoS depth
        )
        self.subscription

        self.get_logger().info(
            "DepthImageInspector 節點已啟動，正在訂閱 /depth_image_topic..."
        )

    def listener_callback(self, msg: Image):
        """
        接收到影像訊息後的回調函數。
        解析影像的位元深度並輸出。
        """
        encoding = msg.encoding
        bits_per_pixel = self.get_bits_per_pixel(encoding)

        if bits_per_pixel:
            self.get_logger().info(
                f"接收到影像：{msg.width}x{msg.height}, Encoding: {encoding}, Bits per pixel: {bits_per_pixel}"
            )
            try:
                depth_array = self.convert_image_to_array(msg, encoding)
                max_value = np.max(depth_array)
                self.get_logger().info(f"影像中的最大深度值為: {max_value}")
            except ValueError as e:
                self.get_logger().error(f"解析影像數據時發生錯誤: {e}")
        else:
            self.get_logger().warn(f"接收到影像，但無法解析 Encoding: {encoding}")

    @staticmethod
    def get_bits_per_pixel(encoding):
        """
        根據影像的 encoding 解析每個像素的位元深度。
        支援常見的 ROS 2 sensor_msgs/Image encoding。
        """
        encoding_to_bpp = {
            "mono8": 8,
            "mono16": 16,
            "rgb8": 24,
            "rgba8": 32,
            "bgr8": 24,
            "bgra8": 32,
            "mono32": 32,
            "bgr16": 48,
            "rgb16": 48,
            "bgra16": 64,
            "rgba16": 64,
            # 可根據需要擴充其他 encoding
        }
        return encoding_to_bpp.get(encoding, None)

    @staticmethod
    def convert_image_to_array(msg, encoding):
        """
        將 ROS Image 訊息轉換為 NumPy 陣列，並根據 encoding 解碼。
        """
        if encoding == "mono8":
            return np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width
            )
        elif encoding == "mono16":
            return np.frombuffer(msg.data, dtype=np.uint16).reshape(
                msg.height, msg.width
            )
        elif encoding == "32FC1":
            return np.frombuffer(msg.data, dtype=np.float32).reshape(
                msg.height, msg.width
            )
        else:
            raise ValueError(f"Unsupported encoding: {encoding}")


def main(args=None):
    rclpy.init(args=args)
    node = DepthImageInspector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("收到 KeyboardInterrupt，正在關閉節點...")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
import cv2
import numpy as np

# 顏色定義
BLUE = (255, 0, 0)  # BGR
BROWN = (19, 69, 139)  # BGR
TOLERANCE = 30

class DoorDetectorNode(Node):
    def __init__(self):
        super().__init__('door_detector')
        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/image/compressed',
            self.image_callback,
            10)
        self.publisher_ = self.create_publisher(Image, '/door_detector/debug_image', 10)
        self.bridge = CvBridge()
        self.get_logger().info("Door Detector Node Initialized.")

    def image_callback(self, msg):
        cv_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="bgr8")

        # 掃描線位置（中間）
        y = cv_image.shape[0] // 2
        scan_line = cv_image[y]

        def is_color(pixel, target):
            return all(abs(int(p) - int(t)) < TOLERANCE for p, t in zip(pixel, target))

        # 找藍色外框區段
        start_idx, end_idx = None, None
        for i, pixel in enumerate(scan_line):
            if is_color(pixel, BLUE):
                if start_idx is None:
                    start_idx = i
                end_idx = i

        if start_idx is None or end_idx is None:
            self.get_logger().warn("找不到藍色外框")
            return

        section = scan_line[start_idx:end_idx + 1]
        section_length = len(section)
        num_doors = 3
        door_width = section_length // (num_doors * 2)

        door_states = []
        for i in range(num_doors):
            door_center = int((2 * i + 1) * door_width)
            pixel = section[door_center]
            if is_color(pixel, BROWN):
                door_states.append("CLOSED")
                color = (0, 0, 255)  # 紅
            else:
                door_states.append("OPEN")
                color = (0, 255, 0)  # 綠
            # 畫門位置
            cv2.circle(cv_image, (start_idx + door_center, y), 5, color, -1)

        # 顯示狀態
        self.get_logger().info(f"門狀態: {door_states}")

        # 發佈 debug 圖像
        img_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
        self.publisher_.publish(img_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DoorDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

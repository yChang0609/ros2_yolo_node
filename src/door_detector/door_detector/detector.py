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
        self.image_queue = deque(maxlen=5)  
        self.publisher_ = self.create_publisher(Image, '/door_detector/debug_image', 10)
        self.bridge = CvBridge()
        self.get_logger().info("Door Detector Node Initialized.")

        self.create_timer(0.05, self.process_image_queue) 

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
            print("[Error] cv_image is None")
            return
        
        # TODO 
        # 檢查 aruco 之間是否有門 
        # aruco list door1[ 0, 1, 2] door2[ 0, 1, 2] door3[ 0, 1, 2]


def main(args=None):
    rclpy.init(args=args)
    node = DoorDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

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
from sensor_msgs.msg import CompressedImage, Image
from interfaces_pkg.srv import PikachuDetect  
from geometry_msgs.msg import Point

# >> Self package
from yolo_pkg.load_params import LoadParams

class YOLODetectorNode(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        self.debug = True
        self.bridge = CvBridge()
        self.params = LoadParams()
        self.yolo_model = self.params.get_detection_model()
        self.names = self.yolo_model.names
        self.conf = 0.8
        print(self.names)
        # >> image proccess
        self.image_queue = deque(maxlen=5)
        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/image/compressed',
            self.image_callback, 10)
        self.image_pub = self.create_publisher(Image, '/yolo_detector/detected_image', 10)
        self.create_timer(0.05, self.process_image_queue) 

        self.detect_service = self.create_service(
            PikachuDetect, 'detect_pikachu', self.handle_detect_pikachu
        )
        # >> for keyboard input
        threading.Thread(target=self.key_listener, daemon=True).start()

    def key_listener(self):
        import sys
        import termios
        import tty
        import select
        import rclpy

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while rclpy.ok():
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    key = sys.stdin.read(1)
                    if key.lower() == 'q':
                        rclpy.shutdown()
                        break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def image_callback(self, msg):
        if self.yolo_model:
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
        
        results = self.yolo_model(cv_image, verbose=False)
        result = results[0]
        boxes = result.boxes

        for box in boxes:
            cls_id = int(box.cls[0].item())
            label = self.names[cls_id]
            conf = box.conf[0].item()

            if label.lower() != "pikachu" or conf < self.conf:
                continue

            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(cv_image, f"{label} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
        drawn_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
        self.image_pub.publish(drawn_msg)

    def handle_detect_pikachu(self, request, response):
        try:
            # Convert image to cv2
            cv_image = self.bridge.imgmsg_to_cv2(request.image, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"Image conversion failed: {e}")
            response.detected = False
            response.position = Point()
            return response

        results = self.yolo_model(cv_image, verbose=False)
        result = results[0]
        boxes = result.boxes

        for box in boxes:
            cls_id = int(box.cls[0].item())
            label = self.names[cls_id]
            conf = box.conf[0].item()
            if label.lower() == "pikachu" and conf > self.conf:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                cx = float((x1 + x2) / 2.0)
                cy = float((y1 + y2) / 2.0)
                response.detected = True
                response.position = Point(x=cx, y=cy, z=0.0)
                return response

        response.detected = False
        response.position = Point()
        return response

def main(args=None):
    rclpy.init(args=args)
    node = YOLODetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
## >> ROS2
import rclpy
from rclpy.node import Node

# >> Basic package
import cv2
import numpy as np
from collections import deque
from cv_bridge import CvBridge

## >> ROS2 interfaces
from sensor_msgs.msg import CompressedImage, Image


def get_avg_color(img, x, y, dx, dy, side=1, size=3):
    h, w, _ = img.shape
    half = size // 2
    values = []

    for i in range(-half, half+1):
        sx = x + dx * side + dy * i
        sy = y + dy * side + dx * i
        if 0 <= sx < w and 0 <= sy < h:
            values.append(img[sy, sx].astype(int))

    if values:
        return np.mean(values, axis=0)
    return None

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

        self.cluster_centers = np.load("/workspaces/src/door_detector/kmeans_models/kmeans_centers.npy")
        self.get_logger().info("Load KMeans cluster_centers (npy)")

        self.label_map = {
            0: "wall",
            1: "pillar",
            2: "door",
            3: "floor",
            4: "other",
        }

        self.create_timer(0.05, self.process_image_queue) 

    def classify_color(self, color):
        distances = np.linalg.norm(self.cluster_centers - color, axis=1)
        return int(np.argmin(distances))
    
    def image_callback(self, msg):
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
        

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
        dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)


        # height, width = edges.shape
        edge_color_image = cv_image.copy()
        
        ys, xs = np.where(edges == 255)
        for x, y in zip(xs, ys):
            gx = dx[y, x]
            gy = dy[y, x]
            norm = np.sqrt(gx**2 + gy**2)
            if norm == 0:
                continue
            nx = int(round(gx / norm))
            ny = int(round(gy / norm))

            angle = np.abs(np.arctan2(gy, gx) * 180.0 / np.pi)
            if angle < 20 or angle > 100:
                continue

            try:
                c1 = get_avg_color(cv_image, x, y, nx, ny, side=40)
                c2 = get_avg_color(cv_image, x, y, nx, ny, side=-40)
            except IndexError:
                continue

            if c1 is None or c2 is None:
                continue

            label1 = self.label_map[self.classify_color(c1)]
            label2 = self.label_map[self.classify_color(c2)]

            labels = [label1, label2]



            if "floor" in labels:
                if "door" in labels:
                    edge_color_image[y, x] = (255, 0, 0)
                elif "wall" in labels:
                    edge_color_image[y, x] = (0, 0, 255)
                elif "pillar" in labels:
                    edge_color_image[y, x] = (0, 255, 0)
                elif "other" in labels:
                    edge_color_image[y, x] = (125, 125, 125)

        debug_msg = self.bridge.cv2_to_imgmsg(edge_color_image, encoding="bgr8")
        self.publisher_.publish(debug_msg)

def main(args=None):
    rclpy.init(args=args)
    node = DoorDetectorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

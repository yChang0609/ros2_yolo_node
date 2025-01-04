import rclpy
from yolo_pkg.core.ros_communicator import RosCommunicator
from yolo_pkg.core.image_processor import ImageProcessor
from yolo_pkg.core.yolo_depth_extractor import YoloDepthExtractor
from yolo_pkg.core.yolo_bounding_box import YoloBoundingBox
from yolo_pkg.core.boundingbox_visaulizer import BoundingBoxVisualizer
from yolo_pkg.core.camera_geometry import CameraGeometry
import threading

def _init_ros_node():
    rclpy.init()
    node = RosCommunicator()
    thread = threading.Thread(target=rclpy.spin, args=(node,))
    thread.start()
    return node, thread

def main():
    ros_communicator, ros_thread = _init_ros_node()
    image_processor = ImageProcessor(ros_communicator)
    yolo_boundingbox = YoloBoundingBox(image_processor, ros_communicator)
    yolo_depth_extractor = YoloDepthExtractor(yolo_boundingbox, image_processor, ros_communicator)
    boundingbox_visualizer = BoundingBoxVisualizer(image_processor, yolo_boundingbox, ros_communicator)
    camera_geometry = CameraGeometry(yolo_depth_extractor)
    
    while(1):
        boundingbox_visualizer.draw_bounding_boxes(screenshot_mode=False, draw_crosshair=True)
        

if __name__ == '__main__':
    main()
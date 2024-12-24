import rclpy
from yolo_pkg.ros_communicator import RosCommunicator
from yolo_pkg.boundingbox_visaulizer import BoundingBoxVisualizer
from yolo_pkg.yolo_detect_model import YoloDetectionModel
from yolo_pkg.object_detect_manager import ObjectDetectManager
import threading

def init_ros_node():
    rclpy.init()
    node = RosCommunicator()
    thread = threading.Thread(target=rclpy.spin, args=(node,))
    thread.start()
    return node, thread

def main():
    ros_communicator, ros_thread = init_ros_node()
    yolo_model = YoloDetectionModel()
    object_detect_manager = ObjectDetectManager(ros_communicator, yolo_model)
    boundingbox_visualizer = BoundingBoxVisualizer(ros_communicator, object_detect_manager)
    
    while(1):
        boundingbox_visualizer.draw_bounding_boxes()

if __name__ == '__main__':
    main()
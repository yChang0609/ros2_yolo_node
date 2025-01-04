import rclpy
from rclpy.executors import MultiThreadedExecutor
from yolo_pkg.core.ros_communicator import RosCommunicator
from yolo_pkg.core.image_processor import ImageProcessor
from yolo_pkg.core.yolo_depth_extractor import YoloDepthExtractor
from yolo_pkg.core.yolo_bounding_box import YoloBoundingBox
from yolo_pkg.core.boundingbox_visaulizer import BoundingBoxVisualizer
from yolo_pkg.core.camera_geometry import CameraGeometry
import threading


def _init_ros_node():
    """
    Initialize the ROS 2 node with MultiThreadedExecutor for efficient handling of multiple subscribers.
    """
    rclpy.init()
    node = RosCommunicator()  # Initialize the ROS node
    executor = MultiThreadedExecutor()  # Use MultiThreadedExecutor
    executor.add_node(node)  # Add the node to the executor
    thread = threading.Thread(
        target=executor.spin
    )  # Start the executor in a separate thread
    thread.start()
    return node, executor, thread  # Return the node, executor, and thread


def main():
    """
    Main function to initialize the node and run the bounding box visualizer.
    """
    ros_communicator, executor, ros_thread = _init_ros_node()
    image_processor = ImageProcessor(ros_communicator)
    yolo_boundingbox = YoloBoundingBox(image_processor)
    yolo_depth_extractor = YoloDepthExtractor(
        yolo_boundingbox, image_processor, ros_communicator
    )
    boundingbox_visualizer = BoundingBoxVisualizer(
        image_processor, yolo_boundingbox, ros_communicator
    )
    camera_geometry = CameraGeometry(yolo_depth_extractor)

    try:
        while True:
            boundingbox_visualizer.draw_bounding_boxes(
                screenshot=False, draw_crosshair=True, save_folder="screenshots"
            )
            a = yolo_depth_extractor.get_yolo_object_depth()
            print(a)
    except KeyboardInterrupt:
        print("Shutting down gracefully...")
    finally:
        # Shut down the executor and ROS
        executor.shutdown()
        rclpy.shutdown()
        ros_thread.join()


if __name__ == "__main__":
    main()

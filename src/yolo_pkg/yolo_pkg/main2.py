import rclpy
from rclpy.executors import MultiThreadedExecutor
from yolo_pkg.ros_communicator import RosCommunicator
from yolo_pkg.image_processor import ImageProcessor
from yolo_pkg.yolo_depth_extractor import YoloDepthExtractor
from yolo_pkg.yolo_bounding_box import YoloBoundingBox
from yolo_pkg.boundingbox_visaulizer import BoundingBoxVisualizer
from yolo_pkg.camera_geometry import CameraGeometry
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


def menu():
    print("Select mode:")
    print("1: Draw bounding boxes without screenshot.")
    print("2: Draw bounding boxes with screenshot.")
    print("Press Ctrl+C to exit.")

    user_input = input("Enter your choice (1/2): ")
    return user_input


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

    user_input = menu()

    try:
        while True:

            if user_input == "1":
                boundingbox_visualizer.draw_bounding_boxes(
                    screenshot=False, draw_crosshair=True, save_folder="screenshots"
                )
            elif user_input == "2":
                boundingbox_visualizer.draw_bounding_boxes(
                    screenshot=True, draw_crosshair=True, save_folder="screenshots"
                )
            else:
                print("Invalid input. Please enter 1 or 2.")

            # Example action for yolo_depth_extractor (can be removed if not needed)
            depth_data = yolo_depth_extractor.get_yolo_object_depth()
            print(f"Object Depth: {depth_data}")

    except KeyboardInterrupt:
        print("Shutting down gracefully...")
    finally:
        # Shut down the executor and ROS
        executor.shutdown()
        rclpy.shutdown()
        ros_thread.join()


if __name__ == "__main__":
    main()

# ros2_yolo_integration
## yolo_pkg
### Usage
1. Run the provided activation script to start the container and prepare the environment
```
./yolo_activate.sh
```
2. Do colcon build and source ./install/setup.bash
```
r
```
3. Run yolo node
```
ros2 run yolo_pkg yolo_detection_node
```
### class diagram
![Logo](https://github.com/alianlbj23/ros2_yolo_integration/blob/dev/img/image_deal.jpeg?raw=true)
## yolo_example_pkg
This is a ROS 2 project for integrating YOLO with ROS 2, providing functionality for real-time object detection and bounding box visualization.
### Features
- Receive compressed images from the /yolo/detection/compressed topic.
- Process images to convert them into OpenCV format.
- Draw bounding boxes around detected objects on the images.
- Publish processed images back to the /yolo/detection/compressed topic, which includes the drawn bounding boxes.
### Usage
1. Run the provided activation script to start the container and prepare the environment
```
./yolo_activate.sh
```
2. Do colcon build and source ./install/setup.bash
```
r
```
3. Run yolo node
```
ros2 run yolo_example_pkg yolo_node
```

## How to put your yolo model
To use the YOLO model in this package, follow these steps:
1. Place the Model File

    Download or obtain the YOLO .pt file (e.g., yolov8n.pt) and place it inside the models directory within the pkg folder:

    ```
    <your_ros2_workspace>/
    ├── src/
    │   ├── yolo_example_pkg/
    │   │   ├── models/
    │   │   │   ├── yolov8n.pt
    ```


2. Model Path Configuration

    The script dynamically loads the model from the package's shared directory using the following code:

    ```
    import os
    from ament_index_python.packages import get_package_share_directory

    model_path = os.path.join(
        get_package_share_directory("yolo_example_pkg"), "models", "yolov8n.pt"
    )
    ```
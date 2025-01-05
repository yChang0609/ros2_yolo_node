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
ros2 run yolo_example_pkg yolo_node
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

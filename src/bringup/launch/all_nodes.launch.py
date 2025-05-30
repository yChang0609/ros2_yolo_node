from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='aruco_detector',
            executable='pose_estimator',
            name='aruco_estimator',
            output='screen'
        ),
        Node(
            package='yolo_pkg',
            executable='yolo_detector',
            name='yolo',
            # output='screen'
        ),
    ])
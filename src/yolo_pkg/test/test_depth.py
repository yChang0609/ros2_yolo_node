from sensor_msgs.msg import Image
import numpy as np
from unittest.mock import MagicMock
from yolo_pkg.core.image_processor import ImageProcessor

depth_msg = Image()
depth_msg.encoding = "16UC1"
depth_msg.height = 480
depth_msg.width = 640
depth_msg.step = depth_msg.width * 2
depth_msg.data = np.random.randint(
    0, 5000, (depth_msg.height, depth_msg.width), dtype=np.uint16
).tobytes()

image_processor = ImageProcessor(ros_communicator=MagicMock())
depth_image = image_processor._convert_image_from_ros_to_cv(depth_msg, mode="depth")

print("Depth image shape:", depth_image.shape)
print("Depth image sample (meters):", depth_image[240, 320])

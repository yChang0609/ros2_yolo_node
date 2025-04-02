import struct
import numpy as np
import cv2
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge


class ImageProcessor:
    def __init__(self, ros_communicator):
        self.bridge = CvBridge()
        self.ros_communicator = ros_communicator

    def _convert_imgmsg_to_cv2(self, img, encoding):
        return self.bridge.imgmsg_to_cv2(img, desired_encoding=encoding)

    def _decode_compressed_depth(self, img_msg):
        """
        Decodes a ROS CompressedImage with format '16UC1; compressedDepth'.

        This function checks if the data starts with the PNG signature.
        If it does, it decodes the PNG directly.
        Otherwise, it assumes a custom header (8 bytes: two floats) is present.
        """
        png_signature = bytes([137, 80, 78, 71, 13, 10, 26, 10])
        data_bytes = bytes(img_msg.data)

        if data_bytes.startswith(png_signature):
            # No header present; decode PNG directly.
            png_data = data_bytes
            depth_quant_a = 1.0  # Default scaling factor
            depth_quant_b = 0.0  # Default offset
        else:
            # If there's a custom header, extract it.
            if len(data_bytes) < 8:
                raise ValueError("Compressed depth message is too short!")
            header = data_bytes[:8]
            depth_quant_a, depth_quant_b = struct.unpack("ff", header)
            png_data = data_bytes[8:]

        np_arr = np.frombuffer(png_data, np.uint8)
        depth_png = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        if depth_png is None:
            raise ValueError("Failed to decode PNG depth image")

        # If a header was provided, use it to compute real depth.
        # Otherwise, assume the PNG directly encodes 16UC1 data in millimeters.
        if depth_quant_a != 1.0 or depth_quant_b != 0.0:
            depth_float = np.zeros_like(depth_png, dtype=np.float32)
            valid = depth_png > 0
            depth_float[valid] = depth_quant_a / (
                depth_png[valid].astype(np.float32) - depth_quant_b
            )
        else:
            # Convert from uint16 (assumed in millimeters) to float32 in meters.
            depth_float = depth_png.astype(np.float32) / 1000.0

        return depth_float

    def _convert_image_from_ros_to_cv(self, img, mode):
        """Converts ROS image to OpenCV format (np.ndarray)."""
        try:
            if isinstance(img, CompressedImage):
                if img.format.startswith("16UC1; compressedDepth"):
                    cv_image = self._decode_compressed_depth(img)
                else:
                    cv_image = self.bridge.compressed_imgmsg_to_cv2(img)
            elif isinstance(img, Image):
                encoding = "bgr8" if mode == "rgb" else "16UC1"
                cv_image = self._convert_imgmsg_to_cv2(img, encoding)
                if encoding == "16UC1":
                    cv_image = cv_image.astype(np.float32) / 1000.0
            else:
                raise TypeError("Unsupported image type.")

            if not isinstance(cv_image, np.ndarray):
                raise TypeError("Converted image is not a valid numpy array.")

            return cv_image

        except Exception as e:
            print(f"Error converting image: {e}")
            return None

    def _convert_image_from_cv_to_ros(self, img):
        return self.bridge.cv2_to_compressed_imgmsg(img)

    def get_yolo_target_label(self):
        return self.ros_communicator.get_latest_data("target_label")

    def get_rgb_ros_image(self, img):
        ros_img = self._convert_image_from_cv_to_ros(img)
        return ros_img

    def get_depth_cv_image(self):
        """Fetch and convert the depth image from ROS to OpenCV format."""
        image = self.ros_communicator.get_latest_data("depth_image")
        return self._convert_image_from_ros_to_cv(image, mode="depth")

    def get_rgb_cv_image(self):
        image = self.ros_communicator.get_latest_data("rgb_compress")
        return self._convert_image_from_ros_to_cv(image, mode="rgb")

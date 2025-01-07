import unittest
from unittest.mock import MagicMock
from sensor_msgs.msg import CompressedImage, Image
import numpy as np
from cv_bridge import CvBridge
from yolo_pkg.core.image_processor import ImageProcessor


class TestImageProcessor(unittest.TestCase):
    def setUp(self):
        """Set up mocks and the ImageProcessor instance."""
        # Mock the ROS communicator
        self.mock_ros_communicator = MagicMock()
        # Create an instance of ImageProcessor
        self.image_processor = ImageProcessor(self.mock_ros_communicator)

    def test_convert_image_from_ros_to_cv_compressed_image(self):
        """Test conversion of CompressedImage to OpenCV format."""
        # Arrange
        mock_img = CompressedImage()
        mock_img.data = b"fake_image_data"  # Mock image data
        self.image_processor.bridge.compressed_imgmsg_to_cv2 = MagicMock(
            return_value=np.zeros((480, 640, 3))  # Simulate an RGB image
        )

        # Act
        result = self.image_processor._convert_image_from_ros_to_cv(
            mock_img, mode="rgb"
        )

        # Assert
        self.image_processor.bridge.compressed_imgmsg_to_cv2.assert_called_once_with(
            mock_img
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (480, 640, 3))

    def test_convert_image_from_ros_to_cv_image(self):
        """Test conversion of Image to OpenCV format."""
        # Arrange
        mock_img = Image()
        self.image_processor.bridge.imgmsg_to_cv2 = MagicMock(
            return_value=np.zeros((480, 640))  # Simulate a depth image
        )

        # Act
        result = self.image_processor._convert_image_from_ros_to_cv(
            mock_img, mode="depth"
        )

        # Assert
        self.image_processor.bridge.imgmsg_to_cv2.assert_called_once_with(
            mock_img, desired_encoding="16UC1"
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (480, 640))

    def test_get_depth_cv_image(self):
        """Test fetching and converting depth image."""
        # Arrange
        mock_img = Image()
        self.mock_ros_communicator.get_latest_data = MagicMock(return_value=mock_img)
        self.image_processor.bridge.imgmsg_to_cv2 = MagicMock(
            return_value=np.zeros((480, 640))  # Simulate a depth image
        )

        # Act
        result = self.image_processor.get_depth_cv_image()

        # Assert
        self.mock_ros_communicator.get_latest_data.assert_called_once_with(
            "depth_image_raw"
        )
        self.image_processor.bridge.imgmsg_to_cv2.assert_called_once_with(
            mock_img, desired_encoding="16UC1"
        )
        self.assertIsInstance(result, np.ndarray)

    def test_get_rgb_cv_image(self):
        """Test fetching and converting RGB image."""
        # Arrange
        mock_img = CompressedImage()
        self.mock_ros_communicator.get_latest_data = MagicMock(return_value=mock_img)
        self.image_processor.bridge.compressed_imgmsg_to_cv2 = MagicMock(
            return_value=np.zeros((480, 640, 3))  # Simulate an RGB image
        )

        # Act
        result = self.image_processor.get_rgb_cv_image()

        # Assert
        self.mock_ros_communicator.get_latest_data.assert_called_once_with(
            "rgb_compress"
        )
        self.image_processor.bridge.compressed_imgmsg_to_cv2.assert_called_once_with(
            mock_img
        )
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (480, 640, 3))

    def test_get_yolo_target_label(self):
        """Test fetching YOLO target label."""
        # Arrange
        expected_label = "target_object"
        self.mock_ros_communicator.get_latest_data = MagicMock(
            return_value=expected_label
        )

        # Act
        result = self.image_processor.get_yolo_target_label()

        # Assert
        self.mock_ros_communicator.get_latest_data.assert_called_once_with(
            "target_label"
        )
        self.assertEqual(result, expected_label)


if __name__ == "__main__":
    unittest.main()

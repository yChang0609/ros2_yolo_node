import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2
from datetime import datetime
import os
from yolo_pkg.core.boundingbox_visaulizer import BoundingBoxVisualizer


class TestBoundingBoxVisualizer(unittest.TestCase):
    def setUp(self):
        """Set up mocks and the BoundingBoxVisualizer instance."""
        # Mock dependencies
        self.mock_image_processor = MagicMock()
        self.mock_yolo_bounding_box = MagicMock()
        self.mock_ros_communicator = MagicMock()

        # Create an instance of BoundingBoxVisualizer
        self.visualizer = BoundingBoxVisualizer(
            self.mock_image_processor,
            self.mock_yolo_bounding_box,
            self.mock_ros_communicator,
        )

    @patch("cv2.rectangle")
    @patch("cv2.putText")
    def test_draw_bounding_boxes(self, mock_put_text, mock_rectangle):
        """Test drawing bounding boxes on the image."""
        # Arrange
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)  # Simulate an RGB image
        self.mock_image_processor.get_rgb_cv_image.return_value = mock_image

        detected_objects = [
            {"label": "object1", "confidence": 0.95, "box": (50, 50, 150, 150)},
            {"label": "object2", "confidence": 0.88, "box": (200, 200, 300, 300)},
        ]
        self.mock_yolo_bounding_box.get_tags_and_boxes.return_value = detected_objects

        # Act
        self.visualizer.draw_bounding_boxes(draw_crosshair=False, screenshot=False)

        # Assert
        self.mock_image_processor.get_rgb_cv_image.assert_called_once()
        self.mock_yolo_bounding_box.get_tags_and_boxes.assert_called_once()

        # Verify bounding boxes are drawn
        mock_rectangle.assert_any_call(mock_image, (50, 50), (150, 150), (0, 255, 0), 2)
        mock_rectangle.assert_any_call(
            mock_image, (200, 200), (300, 300), (0, 255, 0), 2
        )

        # Verify labels are drawn
        mock_put_text.assert_any_call(
            mock_image,
            "object1 (0.95)",
            (50, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
        mock_put_text.assert_any_call(
            mock_image,
            "object2 (0.88)",
            (200, 190),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    @patch("cv2.imwrite")
    def test_draw_bounding_boxes_with_screenshot(self, mock_imwrite):
        """Test drawing bounding boxes with screenshot saving."""
        # Arrange
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.mock_image_processor.get_rgb_cv_image.return_value = mock_image

        detected_objects = [
            {"label": "object1", "confidence": 0.95, "box": (50, 50, 150, 150)},
        ]
        self.mock_yolo_bounding_box.get_tags_and_boxes.return_value = detected_objects

        # Act
        with patch("datetime.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2025, 1, 1, 12, 0, 0)
            self.visualizer.draw_bounding_boxes(draw_crosshair=False, screenshot=True)

        # Assert
        mock_imwrite.assert_called_once_with(
            os.path.join("screenshots", "object1_20250101_120000.png"),
            mock_image[50:150, 50:150],
        )

    @patch("cv2.line")
    def test_draw_bounding_boxes_with_crosshair(self, mock_line):
        """Test drawing bounding boxes with crosshair."""
        # Arrange
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.mock_image_processor.get_rgb_cv_image.return_value = mock_image
        self.mock_yolo_bounding_box.get_tags_and_boxes.return_value = []

        # Act
        self.visualizer.draw_bounding_boxes(draw_crosshair=True, screenshot=False)

        # Assert
        # Verify crosshair lines are drawn
        mock_line.assert_any_call(
            mock_image, (320 - 20, 240), (320 + 20, 240), (0, 0, 255), 2
        )
        mock_line.assert_any_call(
            mock_image, (320, 240 - 20), (320, 240 + 20), (0, 0, 255), 2
        )

    def test_draw_bounding_boxes_no_image(self):
        """Test handling when no image is received."""
        # Arrange
        self.mock_image_processor.get_rgb_cv_image.return_value = None

        # Act
        self.visualizer.draw_bounding_boxes(draw_crosshair=False, screenshot=False)

        # Assert
        self.mock_image_processor.get_rgb_cv_image.assert_called_once()
        self.mock_yolo_bounding_box.get_tags_and_boxes.assert_not_called()

    def test_draw_bounding_boxes_publish_image(self):
        """Test publishing processed image to ROS."""
        # Arrange
        mock_image = np.zeros((480, 640, 3), dtype=np.uint8)
        self.mock_image_processor.get_rgb_cv_image.return_value = mock_image
        self.mock_yolo_bounding_box.get_tags_and_boxes.return_value = []

        mock_ros_image = MagicMock()
        self.mock_image_processor.get_rgb_ros_image.return_value = mock_ros_image

        # Act
        self.visualizer.draw_bounding_boxes(draw_crosshair=False, screenshot=False)

        # Assert
        self.mock_image_processor.get_rgb_ros_image.assert_called_once_with(mock_image)
        self.mock_ros_communicator.publish_data.assert_called_once_with(
            "yolo_image", mock_ros_image
        )


if __name__ == "__main__":
    unittest.main()

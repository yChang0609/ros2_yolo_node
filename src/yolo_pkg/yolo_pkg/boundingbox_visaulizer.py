import cv2
import numpy as np
import os
from datetime import datetime
import time
import json


class BoundingBoxVisualizer:
    def __init__(self, image_processor, yolo_bounding_box, ros_communicator):
        self.ros_communicator = ros_communicator
        self.image_processor = image_processor
        self.yolo_bounding_box = yolo_bounding_box
        self.last_screenshot_time = 0
        self.screenshot_interval = 1 / 5

    def _draw_crosshair(self, image):

        height, width = image.shape[:2]

        cx = width // 2
        cy = height // 2

        crosshair_color = (0, 0, 255)
        crosshair_thickness = 2
        crosshair_length = 20

        cv2.line(
            image,
            (cx - crosshair_length, cy),
            (cx + crosshair_length, cy),
            crosshair_color,
            crosshair_thickness,
        )

        cv2.line(
            image,
            (cx, cy - crosshair_length),
            (cx, cy + crosshair_length),
            crosshair_color,
            crosshair_thickness,
        )

    def draw_offset_info(self, offsets_3d_json):
        """
        Draws 3D offset information on the image along with a point at each object's center.
        Each object gets a unique color for both its point and text.

        Args:
            offsets_3d_json (str): JSON string containing object offsets in FLU format
                                  (Forward/depth, Left, Up)

        Returns:
            None
        """
        if not offsets_3d_json:
            return

        # Get the current image
        image = self.image_processor.get_rgb_cv_image()
        if image is None:
            print("Error: No image received from image_processor")
            return

        try:
            # Parse the JSON string
            offsets_data = json.loads(offsets_3d_json)

            if not offsets_data:
                return

            # Find detected objects to match with offsets
            detected_objects = self.yolo_bounding_box.get_tags_and_boxes()
            if not detected_objects:
                return

            # Create a mapping from label to box
            label_to_box = {obj["label"]: obj["box"] for obj in detected_objects}

            # Predefined colors for objects (BGR format)
            colors = [
                (0, 0, 255),  # Red
                (0, 255, 0),  # Green
                (255, 0, 0),  # Blue
                (255, 0, 255),  # Magenta
                (0, 255, 255),  # Yellow
                (255, 165, 0),  # Orange
                (128, 0, 128),  # Purple
                (0, 128, 128),  # Teal
                (128, 128, 0),  # Olive
                (75, 0, 130),  # Indigo
            ]

            # Draw offset information for each object
            for i, offset_obj in enumerate(offsets_data):
                label = offset_obj["label"]
                offset = offset_obj.get("offset_flu")

                if not offset or label not in label_to_box:
                    continue

                x1, y1, x2, y2 = label_to_box[label]

                # Select color based on object index (cycle through colors if more objects than colors)
                color = colors[i % len(colors)]

                # Calculate center point of bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Draw a circle at the center point using the object's color
                cv2.circle(
                    image,
                    (center_x, center_y),
                    5,  # Radius
                    color,  # Object-specific color
                    -1,  # Filled circle
                )

                # Format the offset text: Forward (F), Left (L), Up (U)
                offset_text = f"F:{offset[0]}m L:{offset[1]}m U:{offset[2]}m"

                # Draw the offset below the bounding box using the same color
                text_y = y2 + 20  # Position text below the bounding box
                cv2.putText(
                    image,
                    offset_text,
                    (x1, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,  # Same color as the center point
                    2,
                )

            # Convert to ROS message and publish
            ros_image = self.image_processor.get_rgb_ros_image(image)
            self.ros_communicator.publish_data("yolo_image", ros_image)

        except json.JSONDecodeError:
            print("Error decoding JSON from offsets_3d")
        except Exception as e:
            print(f"Error drawing offset info: {e}")

        # Convert to ROS message and publish
        try:
            ros_image = self.image_processor.get_rgb_ros_image(image)
            self.ros_communicator.publish_data("yolo_image", ros_image)
        except Exception as e:
            print(f"Failed to convert or publish image with offsets: {e}")
        # Use this function can 5fps screenshot

    def save_fps_screenshot(self, save_folder="fps_screenshots"):
        """
        Saves full-frame images at 5 FPS without bounding boxes.
        """
        image = self.image_processor.get_rgb_cv_image()
        if image is None:
            print("Error: No image received from image_processor")
            return

        current_time = time.time()
        if current_time - self.last_screenshot_time < self.screenshot_interval:
            return

        self.last_screenshot_time = current_time

        os.makedirs(save_folder, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        screenshot_path = os.path.join(save_folder, f"full_frame_{timestamp}.png")
        cv2.imwrite(screenshot_path, image)
        # print(f"Saved full-frame screenshot: {screenshot_path}")

    def draw_bounding_boxes(
        self,
        draw_crosshair=False,
        screenshot=False,
        save_folder="screenshots",
        segmentation_status=False,
        bounding_status=False,  # Controls all bounding boxes
    ):
        """
        Draws bounding boxes on the output image from the /yolo/detection/compressed topic.

        Args:
            draw_crosshair (bool): If True, draws a crosshair on the subscribed camera topic image.
            screenshot (bool): If True, captures a screenshot of the objects detected by YOLO.
            save_folder (str): Directory where the screenshots will be saved.
            segmentation_status (bool): Controls whether the segmentation overlay is displayed on the output image.
                             Set to True to enable segmentation; set to False to disable segmentation.
            bounding_status (bool): Controls whether the bounding boxes are displayed on the output image.
                                    Set to True to enable bounding boxes; set to False to disable bounding boxes.

        Returns:
            None
        """

        image = self.image_processor.get_rgb_cv_image()
        if image is None:
            print("Error: No image received from image_processor")
            return

        if segmentation_status:
            self.yolo_bounding_box.get_segmentation_data()
            segmentation_objects = self.yolo_bounding_box.get_segmentation_data()

            for obj in segmentation_objects:
                mask = obj["mask"]
                label = obj["label"]
                x1, y1, x2, y2 = obj["box"]

                # Overlay mask with transparency (mask is still drawn)
                mask_colored = np.zeros_like(image, dtype=np.uint8)
                mask_colored[mask > 0] = (0, 255, 0)  # Green mask
                image = cv2.addWeighted(image, 1, mask_colored, 0.5, 0)

                # **Only draw segmentation bounding box if bounding_status is open**
                if bounding_status == "open":
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        image,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

        # **Check if bounding_status is "open" before drawing YOLO bounding boxes**
        if bounding_status:
            detected_objects = self.yolo_bounding_box.get_tags_and_boxes()
            for obj in detected_objects:
                label = obj["label"]
                confidence = obj["confidence"]
                x1, y1, x2, y2 = obj["box"]

                # Draw Bounding Box
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Draw label and confidence
                label_text = f"{label} ({confidence:.2f})"
                cv2.putText(
                    image,
                    label_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                if screenshot:
                    cropped_image = image[y1:y2, x1:x2]
                    if cropped_image.size > 0:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        screenshot_path = os.path.join(
                            save_folder, f"{label}_{timestamp}.png"
                        )
                        cv2.imwrite(screenshot_path, cropped_image)

        if draw_crosshair:
            self._draw_crosshair(image)

        if not isinstance(image, np.ndarray):
            print("Processed image is not a valid numpy array.")
            return

        try:
            # Convert image to ROS message and publish
            ros_image = self.image_processor.get_rgb_ros_image(image)
            self.ros_communicator.publish_data("yolo_image", ros_image)
        except Exception as e:
            print(f"Failed to convert or publish image: {e}")

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

        # 定义一组预设颜色 (BGR格式)
        self.available_colors = [
            (0, 0, 255),  # 红色
            (0, 255, 0),  # 绿色
            (255, 0, 0),  # 蓝色
            (255, 0, 255),  # 洋红色
            (0, 255, 255),  # 黄色
            (255, 165, 0),  # 橙色
            (128, 0, 128),  # 紫色
            (0, 128, 128),  # 蓝绿色
            (128, 128, 0),  # 橄榄色
            (75, 0, 130),  # 靛蓝色
            (255, 192, 203),  # 粉色
            (210, 105, 30),  # 巧克力色
            (32, 178, 170),  # 浅海洋绿
            (0, 0, 128),  # 海军蓝
            (128, 0, 0),  # 栗色
        ]

        # 初始化颜色映射字典，用于存储每个标签对应的固定颜色
        self.label_color_map = {}
        self.next_color_index = 0

    def _get_color_for_label(self, label):
        """
        为给定的标签返回一个固定的颜色。如果标签之前未见过，则分配一个新颜色。

        Args:
            label (str): 物体标签

        Returns:
            tuple: BGR颜色元组
        """
        # 如果标签已经有分配的颜色，直接返回
        if label in self.label_color_map:
            return self.label_color_map[label]

        # 否则，分配一个新颜色并保存到映射中
        color = self.available_colors[
            self.next_color_index % len(self.available_colors)
        ]
        self.label_color_map[label] = color
        self.next_color_index += 1

        return color

    def _draw_object_offsets(self, image, offsets_3d_json):
        """
        Internal function that draws 3D offset information on the image.

        Args:
            image: CV image to draw upon
            offsets_3d_json (str): JSON string containing object offsets

        Returns:
            bool: True if drawing was successful, False otherwise
        """
        if not offsets_3d_json:
            return False

        try:
            # Parse the JSON string
            offsets_data = json.loads(offsets_3d_json)

            if not offsets_data:
                return False

            # Find detected objects to match with offsets
            detected_objects = self.yolo_bounding_box.get_tags_and_boxes()
            if not detected_objects:
                return False

            # Create a mapping from label to box
            label_to_box = {obj["label"]: obj["box"] for obj in detected_objects}

            # Draw offset information for each object
            for offset_obj in offsets_data:
                label = offset_obj["label"]
                offset = offset_obj.get("offset_flu")

                if not offset or label not in label_to_box:
                    continue

                x1, y1, x2, y2 = label_to_box[label]

                # 为该标签获取固定颜色
                color = self._get_color_for_label(label)

                # Calculate center point of bounding box
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                # Draw a circle at the center point using the object's color
                cv2.circle(
                    image,
                    (center_x, center_y),
                    5,  # Radius
                    color,  # Object's fixed color
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
                    color,  # Same fixed color
                    2,
                )

            return True

        except json.JSONDecodeError:
            print("Error decoding JSON from offsets_3d")
            return False
        except Exception as e:
            print(f"Error drawing offset info: {e}")
            return False

    def draw_offset_info(self, offsets_3d_json):
        """
        Draws 3D offset information on the image and publishes it.
        This is kept for backwards compatibility.

        Args:
            offsets_3d_json (str): JSON string containing object offsets

        Returns:
            None
        """
        # Get the current image
        image = self.image_processor.get_rgb_cv_image()
        if image is None:
            print("Error: No image received from image_processor")
            return

        # Draw offsets on the image
        self._draw_object_offsets(image, offsets_3d_json)

        # Publish the image
        try:
            ros_image = self.image_processor.get_rgb_ros_image(image)
            self.ros_communicator.publish_data("yolo_image", ros_image)
        except Exception as e:
            print(f"Failed to convert or publish image with offsets: {e}")

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
        bounding_status=False,
        offsets_3d_json=None,
    ):
        """
        Draws bounding boxes on the output image from the /yolo/detection/compressed topic.

        Args:
            draw_crosshair (bool): If True, draws a crosshair on the subscribed camera topic image.
            screenshot (bool): If True, captures a screenshot of the objects detected by YOLO.
            save_folder (str): Directory where the screenshots will be saved.
            segmentation_status (bool): Controls whether the segmentation overlay is displayed on the output image.
            bounding_status (bool): Controls whether the bounding boxes are displayed on the output image.
            offsets_3d_json (str): Optional JSON string containing object offsets to display.

        Returns:
            None
        """
        image = self.image_processor.get_rgb_cv_image()
        if image is None:
            print("Error: No image received from image_processor")
            return

        # Process segmentation if enabled
        if segmentation_status:
            self.yolo_bounding_box.get_segmentation_data()
            segmentation_objects = self.yolo_bounding_box.get_segmentation_data()

            for obj in segmentation_objects:
                mask = obj["mask"]
                label = obj["label"]
                x1, y1, x2, y2 = obj["box"]

                # 获取该标签的固定颜色
                color = self._get_color_for_label(label)

                # 使用该标签的固定颜色创建蒙版
                mask_colored = np.zeros_like(image, dtype=np.uint8)
                mask_colored[mask > 0] = color  # 使用标签固定颜色
                image = cv2.addWeighted(image, 1, mask_colored, 0.5, 0)

                # **Only draw segmentation bounding box if bounding_status is open**
                if bounding_status == "open":
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(
                        image,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2,
                    )

        # **Check if bounding_status is "open" before drawing YOLO bounding boxes**
        if bounding_status:
            detected_objects = self.yolo_bounding_box.get_tags_and_boxes()
            for obj in detected_objects:
                label = obj["label"]
                confidence = obj["confidence"]
                x1, y1, x2, y2 = obj["box"]

                # 获取该标签的固定颜色
                color = self._get_color_for_label(label)

                # Draw Bounding Box with fixed color
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

                # Draw label and confidence with fixed color
                label_text = f"{label} ({confidence:.2f})"
                cv2.putText(
                    image,
                    label_text,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
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

        # Draw offsets if provided
        if offsets_3d_json:
            self._draw_object_offsets(image, offsets_3d_json)

        if not isinstance(image, np.ndarray):
            print("Processed image is not a valid numpy array.")
            return

        try:
            # Convert image to ROS message and publish
            ros_image = self.image_processor.get_rgb_ros_image(image)
            self.ros_communicator.publish_data("yolo_image", ros_image)
        except Exception as e:
            print(f"Failed to convert or publish image: {e}")

    def _draw_crosshair(self, image):
        """
        在图像中心绘制十字准星。

        Args:
            image: 要绘制的CV图像

        Returns:
            None
        """
        height, width = image.shape[:2]

        # 计算图像中心
        center_x = width // 2
        center_y = height // 2

        # 十字准星参数
        crosshair_color = (0, 0, 255)  # 红色 (BGR)
        crosshair_thickness = 2
        crosshair_length = 20

        # 绘制水平线
        cv2.line(
            image,
            (center_x - crosshair_length, center_y),
            (center_x + crosshair_length, center_y),
            crosshair_color,
            crosshair_thickness,
        )

        # 绘制垂直线
        cv2.line(
            image,
            (center_x, center_y - crosshair_length),
            (center_x, center_y + crosshair_length),
            crosshair_color,
            crosshair_thickness,
        )

import cv2
import numpy as np
import os
from datetime import datetime
import time


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
        print(f"Saved full-frame screenshot: {screenshot_path}")

    def draw_bounding_boxes(
        self, draw_crosshair=False, screenshot=False, save_folder="screenshots"
    ):
        """
        根據 YOLO 偵測結果在影像上繪製 Bounding Box。
        """
        detected_objects = self.yolo_bounding_box.get_tags_and_boxes()
        image = self.image_processor.get_rgb_cv_image()
        if image is None:
            print("Error: No image received from image_processor")
            return
        # 繪製 Bounding Box
        for obj in detected_objects:
            label = obj["label"]
            confidence = obj["confidence"]
            x1, y1, x2, y2 = obj["box"]

            # 繪製 Bounding Box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 繪製標籤與置信度
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

        if not isinstance(image, (np.ndarray,)):
            print("Processed image is not a valid numpy array.")
            return

        try:
            # 將影像轉換為 ROS 訊息並發布
            ros_image = self.image_processor.get_rgb_ros_image(image)
            self.ros_communicator.publish_data("yolo_image", ros_image)
        except Exception as e:
            print(f"Failed to convert or publish image: {e}")

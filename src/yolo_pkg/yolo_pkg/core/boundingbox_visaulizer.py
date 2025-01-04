import cv2
import numpy as np
import os
import time
from datetime import datetime

class BoundingBoxVisualizer():
    def __init__(self, image_processor, yolo_bounding_box, ros_communicator):
        self.ros_communicator = ros_communicator
        self.image_processor = image_processor
        self.yolo_bounding_box = yolo_bounding_box
    
    def _draw_crosshair(self, image):

        height, width = image.shape[:2]
        
        cx = width // 2
        cy = height // 2

        crosshair_color = (0, 0, 255)  # 紅色
        crosshair_thickness = 2
        crosshair_length = 20  # 十字線長度

        # 水平線
        cv2.line(
            image,
            (cx - crosshair_length, cy),
            (cx + crosshair_length, cy),
            crosshair_color,
            crosshair_thickness
        )

        # 垂直線
        cv2.line(
            image,
            (cx, cy - crosshair_length),
            (cx, cy + crosshair_length),
            crosshair_color,
            crosshair_thickness
        )

    def draw_bounding_boxes(self, draw_crosshair=False, screenshot_mode=False, save_folder="screenshots"):
        """
        根據 YOLO 偵測結果在影像上繪製 Bounding Box。
        """
        # 獲取 RGB 影像
        # 獲取標籤與框座標（包含信心值過濾和目標標籤過濾）
        detected_objects = self.yolo_bounding_box.get_tags_and_boxes()
        image = self.image_processor.get_rgb_cv_image()
        if image is None:
            print("Error: No image received from object_detect_manager.get_cv_image()")
            return
        # 繪製 Bounding Box
        for obj in detected_objects:
            label = obj['label']
            confidence = obj['confidence']
            x1, y1, x2, y2 = obj['box']

            # 繪製 Bounding Box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # 繪製標籤與置信度
            label_text = f"{label} ({confidence:.2f})"
            cv2.putText(
                image, label_text, (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )
            if screenshot_mode:
                cropped_image = image[y1:y2, x1:x2]
                if cropped_image.size > 0:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    screenshot_path = os.path.join(save_folder, f"{label}_{timestamp}.png")
                    cv2.imwrite(screenshot_path, cropped_image)
                    print(f"Saved cropped screenshot for {label} to: {screenshot_path}")
            
        if draw_crosshair:
            self._draw_crosshair(image)

        if not isinstance(image, (np.ndarray,)):
            print("Processed image is not a valid numpy array.")
            return

        try:
            # 將影像轉換為 ROS 訊息並發布
            ros_image = self.image_processor.get_rgb_ros_image(image)
            self.ros_communicator.publish_yolo_image(ros_image)
        except Exception as e:
            print(f"Failed to convert or publish image: {e}")

        
        
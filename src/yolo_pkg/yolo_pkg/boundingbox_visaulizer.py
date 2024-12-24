from cv_bridge import CvBridge
import cv2
import numpy as np

class BoundingBoxVisualizer():
    def __init__(self, ros_communicator, object_detect_manager):
        self.ros_communicator = ros_communicator
        self.object_detect_manager = object_detect_manager
    
    def draw_bounding_boxes(self):
        """
        根據 YOLO 偵測結果在影像上繪製 Bounding Box。
        """
        # 獲取 RGB 影像
        # 獲取標籤與框座標（包含信心值過濾和目標標籤過濾）
        detected_objects = self.object_detect_manager.get_tags_and_boxes()
        image = self.object_detect_manager.get_cv_image()
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

        if not isinstance(image, (np.ndarray,)):
            print("Processed image is not a valid numpy array.")
            return

        try:
            # 將影像轉換為 ROS 訊息並發布
            ros_image = self.object_detect_manager.get_ros_image()
            self.ros_communicator.publish_yolo_image(ros_image)
        except Exception as e:
            print(f"Failed to convert or publish image: {e}")

        
        
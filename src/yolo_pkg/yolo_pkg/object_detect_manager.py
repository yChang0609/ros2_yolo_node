import contextlib
import io
from cv_bridge import CvBridge
import numpy as np

class ObjectDetectManager():
    def __init__(self, ros_communicator, yolo_model):
        self.model = yolo_model.get_model()
        self.ros_communicator = ros_communicator
        self.target_label = None
        self.image = None
        self.bridge = CvBridge()

    def convert_image_to_cv(self):
        """Converts ROS image to OpenCV format (np.ndarray)."""
        try:
            self.image = self.ros_communicator.get_latest_image()
            cv_image = self.bridge.compressed_imgmsg_to_cv2(self.image, desired_encoding='bgr8')
            if not isinstance(cv_image, np.ndarray):
                raise TypeError("Converted image is not a valid numpy array.")
            self.image = cv_image
            return self.image
        except Exception as e:
            print(f"Error converting image: {e}")
            return None


    def detect_objects(self, image):
        with contextlib.redirect_stdout(io.StringIO()):
            results = self.model(image, verbose=False)
        return results

    def get_cv_image(self):
        return self.image

    def get_ros_image(self):
        return self.bridge.cv2_to_compressed_imgmsg(self.image)

    def get_target_label(self):
        target_label = self.ros_communicator.get_latest_target_label()
        if target_label in [None, "None"]:
            target_label = None  # 不過濾標籤
        return target_label

    def get_tags_and_boxes(self, confidence_threshold=0.5):
        self.target_label = self.get_target_label()
        self.convert_image_to_cv()
        if self.image is None:
            return []
        
        detection_results = self.detect_objects(self.image)

        detected_objects = []
        for result in detection_results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                confidence = float(box.conf)
                
                if confidence < confidence_threshold:
                    continue
                
                if self.target_label and class_name != self.target_label:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detected_objects.append({
                    'label': class_name,
                    'confidence': confidence,
                    'box': (x1, y1, x2, y2)
                })

        return detected_objects
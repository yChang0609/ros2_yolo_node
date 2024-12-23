from cv_bridge import CvBridge
import contextlib
import io

class BoundingBoxVisualizer():
    def __init__(self, ros_communicator, yolo_model):
        self.ros_communicator = ros_communicator
        self.yolo_model = yolo_model
    
    def convert_image(self, msg):
        """trans ROS image to cv format"""
        try:
            return self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Could not convert image: {e}")
            return None

    def get_rgb_image(self):
        return self.ros_communicator.get_latest_image()

    def detect_objects(self, image):
        with contextlib.redirect_stdout(io.StringIO()):
            results = self.model(image, verbose=False)
        return results
    
    def get_tags_and_boxes(self, confidence_threshold=0.5):
        target_label = self.get_target_label()
        image = self.get_rgb_image()
        cv_image = self.convert_image(image)
        if cv_image is None:
            self.get_logger().error("Failed to convert image.")
            return []
        
        detection_results = self.detect_objects(cv_image)

        detected_objects = []
        for result in detection_results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = self.model.names[class_id]
                confidence = float(box.conf)
                
                if confidence < confidence_threshold:
                    continue
                
                if target_label and class_name != target_label:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detected_objects.append({
                    'label': class_name,
                    'confidence': confidence,
                    'box': (x1, y1, x2, y2)
                })
        return detected_objects

    def get_target_label(self):
        target_label = self.ros_communicator.get_latest_target_label()
        if target_label in [None, "None"]:
            target_label = None  # 不過濾標籤
        return target_label

    def draw_bounding_boxes(self):
        
        
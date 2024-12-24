import contextlib
import io
from cv_bridge import CvBridge
import numpy as np
from sensor_msgs.msg import CompressedImage, Image, Imu

class ObjectDetectManager():
    def __init__(self, ros_communicator, yolo_model):
        self.model = yolo_model.get_model()
        self.ros_communicator = ros_communicator
        self.target_label = None
        self.image = None
        self.bridge = CvBridge()

    def convert_image_to_cv(self, img, mode):
        """Converts ROS image to OpenCV format (np.ndarray)."""
        try:
            # 檢查影像格式並選擇對應的轉換方法
            if isinstance(img, CompressedImage):
                cv_image = self.bridge.compressed_imgmsg_to_cv2(img)
            elif isinstance(img, Image):
                encoding = 'bgr8' if mode == "rgb" else '16UC1'
                cv_image = self.bridge.imgmsg_to_cv2(img, desired_encoding=encoding)
            else:
                raise TypeError("Unsupported image type. Expected CompressedImage or Image.")

            # 確認轉換後的結果為 numpy 陣列
            if not isinstance(cv_image, np.ndarray):
                raise TypeError("Converted image is not a valid numpy array.")

            return cv_image

        except Exception as e:
            print(f"Error converting image: {e}")
            return None


    
    def get_depth_cv_image(self):
        """Fetch and convert the depth image from ROS to OpenCV format."""
        image = self.ros_communicator.get_latest_depth_image()
        return self.convert_image_to_cv(image, mode="depth")

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



    def get_yolo_object_depth(self):
        depth_cv_image = self.get_depth_cv_image()
        if depth_cv_image is None or not isinstance(depth_cv_image, np.ndarray):
            print("Depth image is invalid.")
            return []
        detected_objects = self.get_tags_and_boxes()
        if not detected_objects:
            print("No detected objects to calculate depth.")
            return []
        
        objects_with_depth = []
        for obj in detected_objects:
            label = obj['label']
            x1, y1, x2, y2 = obj['box']

            # 提取深度图中对应目标区域的深度值
            depth_region = depth_cv_image[y1:y2, x1:x2]

            if depth_region.size == 0:
                print(f"No depth data available for {label}.")
                continue

            # 计算目标区域的平均深度
            mean_depth = np.mean(depth_region[depth_region > 0])

            objects_with_depth.append({
                'label': label,
                'box': (x1, y1, x2, y2),
                'depth': mean_depth
            })

        return objects_with_depth
    


    def get_tags_and_boxes(self, confidence_threshold=0.5):
        self.target_label = self.get_target_label()
        self.image = self.ros_communicator.get_latest_image()
        self.image = self.convert_image_to_cv(img=self.image, mode="rgb")
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
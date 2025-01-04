import numpy as np

class YoloDepthExtractor():
    def __init__(self, yolo_boundingbox, image_processor, ros_communication):
        self.yolo_boundingbox = yolo_boundingbox
        self.ros_communicator = ros_communication
        self.image_processor = image_processor
        
        def get_yolo_object_depth(self):
            depth_cv_image = self.image_processor.get_depth_cv_image()
            if depth_cv_image is None or not isinstance(depth_cv_image, np.ndarray):
                print("Depth image is invalid.")
                return []
            detected_objects = self.yolo_boundingbox.get_tags_and_boxes()
            
            if not detected_objects:
                print("No detected objects to calculate depth.")
                return []
            
            objects_with_depth = []
            for obj in detected_objects:
                label = obj['label']
                x1, y1, x2, y2 = obj['box']

                depth_region = depth_cv_image[y1:y2, x1:x2]

                if depth_region.size == 0:
                    print(f"No depth data available for {label}.")
                    continue

                mean_depth = np.mean(depth_region[depth_region > 0])

                objects_with_depth.append({
                    'label': label,
                    'box': (x1, y1, x2, y2),
                    'depth': mean_depth
                })

            return objects_with_depth
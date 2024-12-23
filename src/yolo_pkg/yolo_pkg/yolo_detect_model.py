import os
from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO

class YoloDetectionModel():
    def __init__(self):
        model_path = os.path.join(
            get_package_share_directory('yolo_pkg'),
            'models',
            'yolov8n.pt'
        )
        self.model = YOLO(model_path)
        self.model.to('cuda')
    
    def get_model(self):
        return self.model
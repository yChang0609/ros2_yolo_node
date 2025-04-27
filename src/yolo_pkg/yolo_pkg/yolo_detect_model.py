import os
import torch
from ament_index_python.packages import get_package_share_directory
from ultralytics import YOLO


class YoloDetectionModel:
    def __init__(self):
        model_path = os.path.join(
            get_package_share_directory("yolo_pkg"),
            "models",
            "best_nano_auto_augu_super_close_fire.pt",
        )
        self.model = YOLO(model_path)
        print()
        print("*" * 10)
        print("yolo bounding : ")
        if torch.cuda.is_available():
            print("CUDA is available. Using GPU.")
            self.model.to("cuda")
        else:
            print("CUDA is not available. Using CPU.")
            self.model.to("cpu")
        print("*" * 10)

    def get_yolo_model(self):
        return self.model

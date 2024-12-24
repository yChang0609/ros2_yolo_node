from ament_index_python.packages import get_package_share_directory
import os
import yaml

class CameraParameters():
    def __init__(self):
        self.camera_intrinsics = self.load_camera_intrinsics()
    
    def load_camera_intrinsics(self):
        config_path = os.path.join(get_package_share_directory('yolo_pkg'), 'config', 'ost.yaml')
        with open(config_path, 'r') as file:
            camera_params = yaml.safe_load(file)
        
        return {
            "fx": camera_params['camera_matrix']['data'][0],
            "fy": camera_params['camera_matrix']['data'][4],
            "cx": camera_params['camera_matrix']['data'][2],
            "cy": camera_params['camera_matrix']['data'][5]
        }
    
    def get_camera_intrinsics(self):
        return self.camera_intrinsics

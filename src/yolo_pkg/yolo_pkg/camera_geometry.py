import numpy as np

class CameraGeometry():
    def __init__(self, camera_parameters, object_detect_manager):
        self.camera_parameters = camera_parameters.get_camera_intrinsics()
        self.object_detect_manager = object_detect_manager

    def calculate_movement_to_center_crosshair(self):
        """
        計算需要的 3D 位移，使機械手臂的末端移動，讓畫面中心十字架對準物體中心點。
        Returns:
            list: 包含每個物體中心對齊的 3D 位移向量，結合深度值計算
        """
        # 相機內參
        fx = self.camera_intrinsics['fx']
        fy = self.camera_intrinsics['fy']
        cx = self.camera_intrinsics['cx']
        cy = self.camera_intrinsics['cy']

        # 獲取 YOLO 偵測物件的深度和位置
        yolo_objects = self.object_detect_manager.get_yolo_object_depth()

        if not yolo_objects:
            print("No YOLO objects detected or depth data unavailable.")
            return []

        # 存儲每個物體的 3D 偏移量
        movement_vectors = []

        for obj in yolo_objects:
            label = obj['label']
            x1, y1, x2, y2 = obj['box']
            depth_value = obj['depth']

            # 計算物體中心點
            object_center_x = (x1 + x2) / 2
            object_center_y = (y1 + y2) / 2

            # 如果深度值無效，跳過該物體
            if depth_value <= 0:
                print(f"Invalid depth for object {label}. Skipping.")
                continue

            # 計算物體相對於畫面中心的偏移量
            x_offset = (object_center_x - cx) * depth_value / fx
            y_offset = (object_center_y - cy) * depth_value / fy
            z_offset = depth_value  # 保留深度值

            # 3D 位移向量
            offset_in_camera_frame = np.array([x_offset, -y_offset, z_offset])
            movement_vectors.append({
                'label': label,
                'movement': offset_in_camera_frame
            })

        return movement_vectors

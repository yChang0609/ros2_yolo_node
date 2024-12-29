import numpy as np

class CameraGeometry():
    def __init__(self, camera_parameters, object_detect_manager):
        self.camera_intrinsics = camera_parameters.get_camera_intrinsics()
        self.object_detect_manager = object_detect_manager

    def calculate_3d_position(self):
        """
        計算每個物體在 IMU 坐標系中的 3D 位置。
        Returns:
            list: 每個物體的 3D 位置向量。
        """
        return self._process_objects(self._calculate_position)

    def _calculate_movement_to_center_crosshair(self):
        """
        計算需要的 3D 位移，使機械手臂的末端移動，讓畫面中心十字架對準物體中心點。
        Returns:
            list: 包含每個物體中心對齊的 3D 位移向量。
        """
        return self._process_objects(self._calculate_offset)

    def _process_objects(self, calculation_fn):
        """
        通用處理函數，計算每個物體的相關 3D 信息。

        Args:
            calculation_fn (function): 用於計算 3D 信息的函數。

        Returns:
            list: 每個物體的計算結果。
        """
        fx = self.camera_intrinsics['fx']
        fy = self.camera_intrinsics['fy']
        cx = self.camera_intrinsics['cx']
        cy = self.camera_intrinsics['cy']

        yolo_objects = self.object_detect_manager.get_yolo_object_depth()

        if not yolo_objects:
            print("No YOLO objects detected or depth data unavailable.")
            return []

        results = []

        for obj in yolo_objects:
            label = obj['label']
            x1, y1, x2, y2 = obj['box']
            depth_value = obj['depth']

            # 如果深度值無效，跳過該物體
            if depth_value <= 0:
                print(f"Invalid depth for object {label}. Skipping.")
                continue

            # 計算物體中心點
            object_center_x = (x1 + x2) / 2
            object_center_y = (y1 + y2) / 2

            # 使用傳入的計算函數處理物體
            result = calculation_fn(object_center_x, object_center_y, depth_value, fx, fy, cx, cy)
            results.append({'label': label, 'movement': result})

        return results

    def _calculate_position(self, x, y, depth, fx, fy, cx, cy):
        """
        計算物體在 IMU 坐標系中的 3D 位置。

        Args:
            x (float): 物體中心的水平座標。
            y (float): 物體中心的垂直座標。
            depth (float): 深度值。
            fx, fy, cx, cy (float): 相機內參數。

        Returns:
            np.ndarray: 物體的 3D 位置向量。
        """
        X = (x - cx) * depth / fx
        Y = (y - cy) * depth / fy
        Z = depth
        return np.array([Z, -X, -Y])

    def _calculate_offset(self, x, y, depth, fx, fy, cx, cy):
        """
        計算物體中心相對於畫面中心的 3D 偏移量。

        Args:
            x (float): 物體中心的水平座標。
            y (float): 物體中心的垂直座標。
            depth (float): 深度值。
            fx, fy, cx, cy (float): 相機內參數。

        Returns:
            np.ndarray: 物體的 3D 偏移向量。
        """
        x_offset = (x - cx) * depth / fx
        y_offset = (y - cy) * depth / fy
        z_offset = depth
        return np.array([x_offset, -y_offset, z_offset])

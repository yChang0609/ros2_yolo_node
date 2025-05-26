# aruco_config.py
import numpy as np
import cv2
import yaml
import os

class ArucoConfig:
    def __init__(self, marker_length=0.34):
        self.marker_length = marker_length
        self.half_size = marker_length / 2

        self.objp = np.array([
            [-self.half_size,  self.half_size, 0],
            [ self.half_size,  self.half_size, 0],
            [ self.half_size, -self.half_size, 0],
            [-self.half_size, -self.half_size, 0]
        ], dtype=np.float32)

        self.camera_matrix = np.array([
            [576.83946  , 0.0       , 319.59192 ],
            [0.         , 577.82786 , 238.89255 ],
            [0.         , 0.        , 1.        ]
        ])
        self.dist_coeffs = np.array([0.001750, -0.003776, -0.000528, -0.000228, 0.000000])

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_params = self._create_detector_params()

        self.T_camera_base = np.eye(4)
        self.T_camera_base[0:3, 0:3] = np.array([
            [0, 0, 1],
            [-1, 0, 0],
            [0, -1, 0]
        ])
        self.T_camera_base[0:3, 3] = [-0.40, 0.0, 0.0]

        R_align = np.array([
            [1, 0,  0],
            [0, 0, -1],
            [0, 1,  0]
        ])
        self.T_align = np.eye(4)
        self.T_align[0:3, 0:3] = np.linalg.inv(R_align)

    def _create_detector_params(self):
        params = cv2.aruco.DetectorParameters_create()
        params.adaptiveThreshWinSizeMin = 5
        params.adaptiveThreshWinSizeMax = 15
        params.adaptiveThreshWinSizeStep = 5
        params.cornerRefinementWinSize = 5
        params.perspectiveRemoveIgnoredMarginPerCell = 0.15
        return params

    def load_marker_map(self, yaml_path):
        with open(yaml_path, 'r') as f:
            return yaml.safe_load(f)

    def reorder_corners(self, marker_id, corners):
        if marker_id not in [3, 4, 5]:
            reorder = [2, 3, 0, 1]
            return corners[:, reorder, :]
        return corners

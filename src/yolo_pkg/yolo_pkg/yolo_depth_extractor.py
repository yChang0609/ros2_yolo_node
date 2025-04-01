import numpy as np


class YoloDepthExtractor:
    def __init__(self, yolo_boundingbox, image_processor, ros_communication):
        self.yolo_boundingbox = yolo_boundingbox
        self.ros_communicator = ros_communication
        self.image_processor = image_processor

    def get_yolo_object_depth(self):
        depth_cv_image = self.image_processor.get_depth_cv_image()
        is_invalid_depth_image = depth_cv_image is None or not isinstance(
            depth_cv_image, np.ndarray
        )
        if is_invalid_depth_image:
            print("Depth image is invalid.")
            return []
        detected_objects = self.yolo_boundingbox.get_tags_and_boxes()

        if not detected_objects:
            print("No detected objects to calculate depth.")
            return []

        objects_with_depth = []
        for obj in detected_objects:
            label = obj["label"]
            x1, y1, x2, y2 = obj["box"]

            depth_region = depth_cv_image[y1:y2, x1:x2]

            if depth_region.size == 0:
                print(f"No depth data available for {label}.")
                continue

            mean_depth = np.mean(depth_region[depth_region > 0])

            objects_with_depth.append(
                {"label": label, "box": (x1, y1, x2, y2), "depth": mean_depth}
            )

        return objects_with_depth

    def get_depth_camera_center_value(self):
        """
        Returns the depth value at the center point of the depth camera,
        along with the center coordinates.

        Returns:
            dict: Contains 'center' (x,y) coordinates and 'depth' value.
                  Returns None if depth image is invalid.
        """
        depth_cv_image = self.image_processor.get_depth_cv_image()
        is_invalid_depth_image = depth_cv_image is None or not isinstance(
            depth_cv_image, np.ndarray
        )
        if is_invalid_depth_image:
            print("Depth image is invalid.")
            return None

        # Calculate center coordinates
        height, width = depth_cv_image.shape[:2]
        center_x = width // 2
        center_y = height // 2

        # Get depth value at center point
        center_depth = depth_cv_image[center_y, center_x]

        # Handle case where center point might have invalid/zero depth
        if center_depth == 0:
            # Try to find nearest non-zero depth value
            window_size = 5
            window = depth_cv_image[
                max(0, center_y - window_size) : min(
                    height, center_y + window_size + 1
                ),
                max(0, center_x - window_size) : min(width, center_x + window_size + 1),
            ]
            non_zero_values = window[window > 0]
            if len(non_zero_values) > 0:
                center_depth = np.mean(non_zero_values)
            else:
                print("No valid depth value found at center point.")
                return None

        return {"center": (center_x, center_y), "depth": float(center_depth)}

from yolo_pkg.yolo_detect_model import YoloDetectionModel
import contextlib
import io


class YoloBoundingBox:
    def __init__(self, image_processor):
        self.image_processor = image_processor
        self.yolo_model = YoloDetectionModel().get_yolo_model()

    def get_tags_and_boxes(self, confidence_threshold=0.6):
        self.target_label = self.get_target_label()
        self.image = self.image_processor.get_rgb_cv_image()
        if self.image is None:
            return []

        detection_results = self._yolo_msg_filter(self.image)

        detected_objects = []
        for result in detection_results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = self.yolo_model.names[class_id]
                confidence = float(box.conf)

                if confidence < confidence_threshold:
                    continue

                if self.target_label and class_name != self.target_label:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detected_objects.append(
                    {
                        "label": class_name,
                        "confidence": confidence,
                        "box": (x1, y1, x2, y2),
                    }
                )

        return detected_objects

    def _yolo_msg_filter(self, img):
        with contextlib.redirect_stdout(io.StringIO()):
            results = self.yolo_model(img, verbose=False)
        return results

    def get_target_label(self):
        target_label = self.image_processor.get_yolo_target_label()
        if target_label in [None, "None"]:
            target_label = None
        return target_label

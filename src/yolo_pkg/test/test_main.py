import pytest
from unittest.mock import MagicMock, patch
import numpy as np
import cv2
from sensor_msgs.msg import CompressedImage
from yolo_pkg.main import YoloDetectionNode
import rclpy


@pytest.fixture
def node():
    rclpy.init()  # 初始化 ROS 2
    node_instance = YoloDetectionNode()
    yield node_instance  # 測試結束後銷毀節點
    node_instance.destroy_node()
    rclpy.shutdown()


def test_convert_image(node):

    # 模擬一個壓縮影像訊息
    compressed_image = CompressedImage()
    compressed_image.format = "jpeg"
    test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    compressed_image.data = cv2.imencode(".jpg", test_image)[1].tobytes()

    # 測試轉換影像
    with patch.object(node.bridge, "compressed_imgmsg_to_cv2", return_value=test_image):
        converted_image = node.convert_image(compressed_image)

    assert isinstance(converted_image, np.ndarray),
    assert converted_image.shape == (100, 100, 3),


def test_detect_objects(node, mocker):
    # 模擬 YOLO 模型
    mock_model = mocker.patch.object(node, "model")
    mock_result = MagicMock()
    mock_result.boxes = [
        MagicMock(cls=np.array([0]), xyxy=np.array([[10, 10, 50, 50]]))
    ]
    mock_model.return_value = [mock_result]

    # 測試偵測
    test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255
    detection_results = node.detect_objects(test_image)

    assert detection_results is not None, "偵測結果不應為 None"
    mock_model.assert_called_once_with(test_image, verbose=False)


def test_publish_image(node, mocker):
    """
    測試 publish_image 方法，檢查影像發布是否成功。
    """
    # 模擬影像數據
    test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255

    # 模擬 ROS publisher
    mock_bridge = mocker.patch.object(node.bridge, "cv2_to_compressed_imgmsg")
    mock_publisher = mocker.patch.object(node.image_pub, "publish")

    # 呼叫 publish_image
    node.publish_image(test_image)

    # 確認是否正確轉換和發布影像
    mock_bridge.assert_called_once_with(test_image)
    mock_publisher.assert_called_once()

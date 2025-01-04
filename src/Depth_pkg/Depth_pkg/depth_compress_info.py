import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import numpy as np
import cv2  # OpenCV 用於解壓縮和處理圖片


class DepthImageAnalyzer(Node):
    def __init__(self):
        super().__init__('depth_image_analyzer')

        # 訂閱 /camera/depth/compressed topic
        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/depth/compressed',
            self.depth_image_callback,
            10
        )

        self.get_logger().info("DepthImageAnalyzer node initialized.")

    def depth_image_callback(self, msg):
        # 解壓縮圖像數據
        try:
            np_arr = np.frombuffer(msg.data, np.uint8)  # 從 msg.data 解碼
            depth_image = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)  # 解壓縮圖像
        except Exception as e:
            self.get_logger().error(f"Failed to decode depth image: {e}")
            return

        # 確認深度圖是否有效
        if depth_image is None:
            self.get_logger().error("Received an empty depth image.")
            return

        # 計算最大值和最小值
        min_val, max_val = np.min(depth_image), np.max(depth_image)

        # 判斷深度圖是幾位元
        depth_bits = depth_image.dtype.itemsize * 8  # 單位是 byte，轉成 bit

        # 輸出結果
        self.get_logger().info(f"Depth Image Stats - Min: {min_val}, Max: {max_val}, Bit Depth: {depth_bits}-bit")


def main(args=None):
    rclpy.init(args=args)

    depth_image_analyzer = DepthImageAnalyzer()

    try:
        rclpy.spin(depth_image_analyzer)
    except KeyboardInterrupt:
        pass
    finally:
        depth_image_analyzer.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

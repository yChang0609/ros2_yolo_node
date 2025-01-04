import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

class DepthTestNode(Node):
    def __init__(self):
        super().__init__('depth_test_node')
        
        # 初始化 CvBridge
        self.bridge = CvBridge()
        
        # 訂閱深度影像主題
        self.depth_sub = self.create_subscription(
            Image,
            '/camera/depth/image_raw',
            self.depth_callback,
            10
        )

    def depth_callback(self, msg):
        """處理深度影像並打印最小深度值"""
        # 將 ROS 影像消息轉換為 OpenCV 格式的深度影像
        depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        
        # 確認影像是否成功轉換
        if depth_image is not None:
            # 計算最小深度值（忽略無效深度）
            min_depth = np.nanmin(depth_image[np.nonzero(depth_image)])  # 過濾掉無效值
            self.get_logger().info(f'Minimum depth value: {min_depth:.3f} meters')
        else:
            self.get_logger().warning("Failed to convert depth image.")

def main(args=None):
    rclpy.init(args=args)
    node = DepthTestNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

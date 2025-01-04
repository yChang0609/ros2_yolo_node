from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image, Imu
from std_msgs.msg import String, Bool
from geometry_msgs.msg import PointStamped

class RosCommunicator(Node):
    def __init__(self):
        super().__init__("RosCommunicator")
        # --- Subscriber Initialization ---

        # Camera Image Subscriber
        self.latest_image = None
        self.image_sub = self.create_subscription(
            CompressedImage,
            '/camera/image/compressed',
            self._image_sub_callback,
            10
        )

        # IMU Data Subscriber
        self.imu_orientation = None
        self.latest_imu = None
        self.imu_sub = self.create_subscription(
            Imu,
            '/imu/data',
            self._imu_sub_callback,
            10
        )

        # Depth Image Subscriber
        self.latest_depth_image = None
        self.depth_image_sub = self.create_subscription(
            CompressedImage,
            '/camera/depth/compressed',
            self._depth_image_sub_callback,
            10
        )
        
        # Target Label Subscriber
        self.latest_target_label = None
        self.target_label_sub = self.create_subscription(
            String,
            '/target_label',
            self._target_label_sub_callback,
            10
        )

        # --- Publisher Initialization ---

        # YOLO Detection Results Publisher
        self.yolo_image_pub = self.create_publisher(CompressedImage, '/yolo/detection/compressed', 10)
        self.point_pub = self.create_publisher(PointStamped, '/yolo/detection/position', 10)
        self.point_offset_pub = self.create_publisher(PointStamped, '/yolo/detection/offset', 10)
        self.detection_status_pub = self.create_publisher(Bool, '/yolo/detection/status', 10)

    # --- Callback Functions ---

    def _image_sub_callback(self, msg):
        self.latest_image = msg

    def _imu_sub_callback(self, msg):
        self.latest_imu = msg
    
    def _target_label_sub_callback(self, msg):
        self.latest_target_label = msg

    def _depth_image_sub_callback(self, msg):
        self.latest_depth_image = msg
    
    # --- Getter Functions ---
    
    def get_latest_rgb_image(self):
        return self.latest_image
    
    def get_latest_depth_image(self):
        return self.latest_depth_image
    
    def get_latest_imu(self):
        return self.latest_imu
    
    def get_latest_target_label(self):
        return self.latest_target_label
    
    
    # --- Publisher Functions ---

    def publish_yolo_image(self, image):
        try:
            self.yolo_image_pub.publish(image)
        except Exception as e:
            self.get_logger().error(f"Could not publish image: {e}")



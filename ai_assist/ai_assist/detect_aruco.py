import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
import cv2 as cv
from cv2 import aruco
import numpy as np
from cv_bridge import CvBridge

class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector')

        # ROS2 Publisher
        self.publisher_ = self.create_publisher(Point, '/detected_marker', 10)

        # ROS2 Subscriber to camera feed
        self.subscription = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, 10)

        # OpenCV Bridge for converting ROS2 images to OpenCV
        self.bridge = CvBridge()

        # Load ArUco dictionary
        self.marker_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
        self.param_markers = aruco.DetectorParameters()

        self.get_logger().info("Aruco Detection Node Started")

    def image_callback(self, msg):
        # Convert ROS2 image to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

        # Convert to grayscale
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Detect ArUco markers
        marker_corners, marker_IDs, _ = aruco.detectMarkers(gray_frame, self.marker_dict, parameters=self.param_markers)

        if marker_corners and marker_IDs is not None:
            for corners in marker_corners:
                flat_corners = corners[0]  # shape: (4, 2)

                center_x = int((flat_corners[0][0] + flat_corners[2][0]) / 2)
                center_y = int((flat_corners[0][1] + flat_corners[2][1]) / 2)

                marker_size = np.linalg.norm(flat_corners[0] - flat_corners[2])
                distance = 500 / marker_size

                frame_width = frame.shape[1]
                normalized_x = (center_x - frame_width // 2) / (frame_width // 2)

                msg = Point()
                msg.x = normalized_x
                msg.y = 0.0
                msg.z = distance
                self.publisher_.publish(msg)

                # ✅ Pass unmodified 'corners' to drawDetectedMarkers
                aruco.drawDetectedMarkers(frame, [corners])
                cv.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)
                cv.putText(frame, f"Dist: {distance:.2f} cm", (center_x, center_y - 10),
                        cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


        # Show frame (Optional)
        cv.imshow("Aruco Detection", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            cv.destroyAllWindows()
            rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()









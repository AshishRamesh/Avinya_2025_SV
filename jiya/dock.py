import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point, Twist
import time

class FollowAruco(Node):
    def __init__(self):
        super().__init__('follow_aruco')

        # Subscribe to detected marker position
        self.subscription = self.create_subscription(
            Point, '/detected_marker', self.listener_callback, 10)

        # Publisher for robot movement
        self.publisher_ = self.create_publisher(Twist, '/cmd_vel', 10)

        # Parameters
        self.declare_parameter("stop_distance", 20.0)  # Stop at 20 cm
        self.stop_distance = self.get_parameter('stop_distance').value

        self.target_x = 0.0
        self.target_z = 1000.0  # Start with a large distance
        self.last_received_time = time.time() - 10000

        self.timer = self.create_timer(0.1, self.timer_callback)

    def timer_callback(self):
        msg = Twist()

        # If marker detected recently
        if (time.time() - self.last_received_time < 1.0):
            self.get_logger().info(f'Target: {self.target_x}, Distance: {self.target_z:.2f} cm')

            if self.target_z > self.stop_distance:
                msg.linear.x = 0.1  # Move forward
            else:
                self.get_logger().info('Reached target distance. Stopping.')
                msg.linear.x = 0.0  # Stop movement

            msg.angular.z = -0.7 * self.target_x  # Rotate to align with marker
        else:
            self.get_logger().info('Target lost. Searching...')
            msg.angular.z = 0.5  # Rotate in place

        self.publisher_.publish(msg)

    def listener_callback(self, msg):
        self.target_x = msg.x
        self.target_z = msg.z
        self.last_received_time = time.time()

def main(args=None):
    rclpy.init(args=args)
    node = FollowAruco()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

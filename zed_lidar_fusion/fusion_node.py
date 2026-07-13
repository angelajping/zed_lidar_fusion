import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from zed_msgs.msg import ObjectsStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from std_msgs.msg import String
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2


class FusionNode(Node):
    def __init__(self):
        super().__init__('fusion_node')

        # Subscribers (time-synced)
        self.lidar_sub  = Subscriber(self, PointCloud2,      '/cx/lslidar_point_cloud')
        self.camera_sub = Subscriber(self, ObjectsStamped,   '/zed/zed_node/obj_det/objects')

        # Time synchronizer
        self.sync = ApproximateTimeSynchronizer(
            [self.lidar_sub, self.camera_sub],
            queue_size=10,
            slop=0.1
        )
        self.sync.registerCallback(self.fusion_callback)

        self.publisher = self.create_publisher(String, '/fused_objects', 10)

        self.get_logger().info('Fusion node started!')

    def fusion_callback(self, lidar_msg, camera_msg):
        points = np.array(list(pc2.read_points(lidar_msg, field_names=('x', 'y', 'z'), skip_nans=True)))

        for obj in camera_msg.objects:
            label = obj.label
            cam_x = obj.position[0]
            cam_y = obj.position[1]
            cam_z = obj.position[2]
            self.get_logger().info(f'Camera sees: {label} at ({cam_x:.2f}, {cam_y:.2f}, {cam_z:.2f})')

            # Find nearest LiDAR point first
            distances = np.sqrt((points[:, 0] - cam_x)**2 + 
                                (points[:, 1] - cam_y)**2 + 
                                (points[:, 2] - cam_z)**2)
            nearest_idx = np.argmin(distances)
            nearest_dist = distances[nearest_idx]

            # Then publish
            msg = String()
            msg.data = f'{label} at {nearest_dist:.2f}m'
            self.publisher.publish(msg)
            self.get_logger().info(f'Published: {msg.data}')


def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
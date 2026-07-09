import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from zed_msgs.msg import ObjectsStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber


class FusionNode(Node):
    def __init__(self):
        super().__init__('fusion_node')

        # TODO: define the static TF2 offset (LiDAR is 0.08m above camera)

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

        # TODO: define a publisher for the fused output

        self.get_logger().info('Fusion node started!')

    def fusion_callback(self, lidar_msg, camera_msg):
        # TODO: convert LiDAR points to camera coordinate frame
        # TODO: cluster LiDAR points into objects
        # TODO: match LiDAR clusters to camera detections by 3D distance
        # TODO: publish fused result
        pass


def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
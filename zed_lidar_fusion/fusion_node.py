import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from zed_msgs.msg import ObjectsStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from std_msgs.msg import String
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
from tf2_ros import Buffer, TransformListener
import tf2_sensor_msgs.tf2_sensor_msgs as tf2_sensor


class FusionNode(Node):
    def __init__(self):
        super().__init__('fusion_node')

        # Transforming 
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Subscribers (time-synced)
        self.lidar_sub  = Subscriber(self, PointCloud2,      '/cx/lslidar_point_cloud')
        self.camera_sub = Subscriber(self, ObjectsStamped,   '/zed/zed_node/obj_det/objects')

        # Time synchronizer
        self.sync = ApproximateTimeSynchronizer(
            [self.lidar_sub, self.camera_sub],
            queue_size=30,
            slop=0.5
        )
        self.sync.registerCallback(self.fusion_callback)

        self.publisher = self.create_publisher(String, '/fused_objects', 10)

        self.get_logger().info('Fusion node started!')

    def fusion_callback(self, lidar_msg, camera_msg):
        #self.get_logger().info('fusion_callback triggered!')
        #pts = pc2.read_points(lidar_msg, field_names=('x', 'y', 'z'), skip_nans=True)
        #points = np.array([[p[0], p[1], p[2]] for p in pts])
        #self.get_logger().info(f'Points shape: {points.shape}, ndim: {points.ndim}')

        try:
            transform = self.tf_buffer.lookup_transform(
                'zed_left_camera_frame',
                lidar_msg.header.frame_id,
                rclpy.time.Time()
            )
            
            # Convert point cloud to list of points, apply full transform manually
            q = transform.transform.rotation
            t = transform.transform.translation
            
            # Build rotation matrix from quaternion
            R = np.array([
                [1 - 2*(q.y**2 + q.z**2),   2*(q.x*q.y - q.z*q.w),   2*(q.x*q.z + q.y*q.w)],
                [2*(q.x*q.y + q.z*q.w),   1 - 2*(q.x**2 + q.z**2),   2*(q.y*q.z - q.x*q.w)],
                [2*(q.x*q.z - q.y*q.w),     2*(q.y*q.z + q.x*q.w), 1 - 2*(q.x**2 + q.y**2)]
            ])
            translation = np.array([t.x, t.y, t.z])
            
            pts = pc2.read_points(lidar_msg, field_names=('x', 'y', 'z'), skip_nans=True)
            raw = np.array([[p[0], p[1], p[2]] for p in pts])
            
            # Apply rotation then translation
            points = (R @ raw.T).T + translation

        except Exception as e:
            self.get_logger().warn(f'TF2 transform failed: {e}')
            return

        if points.ndim != 2 or points.shape[0] == 0:
            self.get_logger().warn('No LiDAR points received, skipping...')
            return

        for obj in camera_msg.objects:
            label = obj.label
            cam_x = obj.position[0]
            cam_y = obj.position[1]
            cam_z = obj.position[2]
            self.get_logger().info(f'Camera sees: {label} at ({cam_x:.2f}, {cam_y:.2f}, {cam_z:.2f})')

            # Find nearest LiDAR point first
            distances = np.abs(points[:, 1] - cam_x)
            nearest_idx = np.argmin(distances)
            nearest_dist = distances[nearest_idx]

            # Then publish
            msg = String()
            msg.data = f'{label} at {nearest_dist:.2f}m'
            self.publisher.publish(msg)
            self.get_logger().info(f'Published: {msg.data}')

            self.get_logger().info(f'Sample LiDAR point: {points[0]}')
            self.get_logger().info(f'Camera position: {cam_x:.2f}, {cam_y:.2f}, {cam_z:.2f}')


def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
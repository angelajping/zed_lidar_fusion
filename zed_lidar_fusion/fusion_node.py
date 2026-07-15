import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from zed_msgs.msg import ObjectsStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from std_msgs.msg import String
import numpy as np
import sensor_msgs_py.point_cloud2 as pc2
from tf2_ros import Buffer, TransformListener
from vision_msgs.msg import Detection3DArray, Detection3D, ObjectHypothesisWithPose

class FusionNode(Node):
    def __init__(self):
        super().__init__('fusion_node')

        # Transforming 
        self.tf_buffer = Buffer() # stores known transforms 
        self.tf_listener = TransformListener(self.tf_buffer, self) # listens for transforms 

        # Subscribers
        self.lidar_sub  = Subscriber(self, PointCloud2,      '/cx/lslidar_point_cloud') 
        self.camera_sub = Subscriber(self, ObjectsStamped,   '/zed/zed_node/obj_det/objects')

        # Time synchronizer -  TODO LOOK INTO IF THIS IS THE BEST OPTION... 
        self.sync = ApproximateTimeSynchronizer(
            [self.lidar_sub, self.camera_sub],
            queue_size=30,
            slop=0.5
        )
        self.sync.registerCallback(self.fusion_callback)

        self.structured_publisher = self.create_publisher(Detection3DArray, '/fused_objects_structured', 10)
        self.string_publisher = self.create_publisher(String, '/fused_objects_display', 10)
        # these hold 10 msgs in a buffer if the subscriber cant keep up

        self.get_logger().info('Fusion node started!') # for debugging purposes

    def fusion_callback(self, lidar_msg, camera_msg):

        try:
            transform = self.tf_buffer.lookup_transform(
                'zed_left_camera_frame', # frame we want to transform to
                lidar_msg.header.frame_id, # frame we're transforming from (laser_link)
                rclpy.time.Time() # use latest available transform
            )
            
            t = transform.transform.translation # pulls three numbers from launch file

            pts = pc2.read_points(lidar_msg, field_names=('x', 'y', 'z'), skip_nans=True)
            raw = np.array([[p[0], p[1], p[2]] for p in pts])
            points = raw + np.array([t.x, t.y, t.z])  # Apply translation to LiDAR points
        
        except Exception as e:
            self.get_logger().warn(f'TF2 transform failed: {e}')
            return

        if points.ndim != 2 or points.shape[0] == 0: # checks if array is 2D and that theres at least one point
            self.get_logger().warn('No LiDAR points received, skipping...') # if so, skips cycle, waits for next msg
            return

        for obj in camera_msg.objects:
            label = obj.label
            cam_x = obj.position[0]
            cam_y = obj.position[1]
            cam_z = obj.position[2]
            # for debugging, what the camera sees the object at in 3D space
            self.get_logger().info(f'Camera sees: {label} at ({cam_x:.2f}, {cam_y:.2f}, {cam_z:.2f})')

            # filtering the points for the lidar to look at 
            y_tolerance = 0.3 # meters left/right
            z_tolerance = 0.3 # meters up/down

            mask = (
                (np.abs(points[:, 1] - cam_y) < y_tolerance) &
                (np.abs(points[:, 2] - cam_z) < z_tolerance)
            )
            nearby_points = points[mask]

            # skip if theres no lidar points nearby the camera object
            if len(nearby_points) == 0:
                self.get_logger().info(f'No nearby LiDAR points for {label}, skipping...')
                continue

            depths = nearby_points[:, 0]  # x axis is forward depth
            nearest_dist = float(np.min(np.abs(depths - cam_x)))  # find nearest distance to camera object

            # String publish for display terminal viewing
            string_msg = String()
            string_msg.data = f'{label} at {nearest_dist:.2f}m\nConfidence Rating: {obj.confidence:.2f}%'
            self.string_publisher.publish(string_msg)
            self.get_logger().info(f'Published: {string_msg.data}')

            # Structured publish for other nodes to use
            detection_array = Detection3DArray()
            detection_array.header = lidar_msg.header

            det = Detection3D()
            det.header = lidar_msg.header

            # adding label and confidence
            hyp = ObjectHypothesisWithPose() # hypothetical object. WithPose means can store position and orientation
            hyp.hypothesis.class_id = label # finding the label of the object
            hyp.hypothesis.score = float(obj.confidence) # the score the camera gives
            det.results.append(hyp)

            # adding 3D position
            det.bbox.center.position.x = cam_x # TODO currently using camera position, but could use nearest lidar point instead
            det.bbox.center.position.y = cam_y
            det.bbox.center.position.z = cam_z
            det.bbox.center.orientation.w = 1.0  # means no rotation

            detection_array.detections.append(det)

            # publishes the detection array! 
            self.structured_publisher.publish(detection_array)


def main(args=None):
    rclpy.init(args=args)
    node = FusionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
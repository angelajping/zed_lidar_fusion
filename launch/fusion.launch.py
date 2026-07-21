from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    zed = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('zed_wrapper'),
            '/launch/zed_camera.launch.py'
        ]),
        launch_arguments={'camera_model': 'zed2i'}.items()
    )

    lidar = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('lslidar_driver'),
            '/launch/lslidar_cx_rviz_launch.py'
        ])
    )

    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments=['-0.0543', '0', '0.08', '-1.57', '0', '0',    # TODO values could be better
                   'zed_camera_center', 'laser_link']
                   # note, used to rotate -1.5708
    )

    fusion = Node(
        package='zed_lidar_fusion',
        executable='fusion_node',
        output='screen'
    )

    return LaunchDescription([zed, lidar, static_tf, fusion])
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument("robot_id", default_value="robot0"),
        DeclareLaunchArgument("image_topic", default_value="/camera/image_raw"),
        DeclareLaunchArgument("camera_info_topic", default_value="/camera/camera_info"),
        DeclareLaunchArgument("network", default_value="dpvo.pth"),
        DeclareLaunchArgument("config", default_value="config/fast.yaml"),
        DeclareLaunchArgument("orb_vocab", default_value="ORBvoc.txt"),
        DeclareLaunchArgument("rerun_connect", default_value=""),
        DeclareLaunchArgument("rerun_recording_id", default_value=""),
        DeclareLaunchArgument("rerun_entity_prefix", default_value=""),
    ]
    node = Node(
        package="dpvo_multi_robot",
        executable="dpvo_multi_robot_node",
        namespace=LaunchConfiguration("robot_id"),
        name="dpvo",
        output="screen",
        parameters=[
            {
                "robot_id": LaunchConfiguration("robot_id"),
                "image_topic": LaunchConfiguration("image_topic"),
                "camera_info_topic": LaunchConfiguration("camera_info_topic"),
                "network": LaunchConfiguration("network"),
                "config": LaunchConfiguration("config"),
                "orb_vocab": LaunchConfiguration("orb_vocab"),
                "rerun_connect": LaunchConfiguration("rerun_connect"),
                "rerun_recording_id": LaunchConfiguration(
                    "rerun_recording_id"
                ),
                "rerun_entity_prefix": LaunchConfiguration(
                    "rerun_entity_prefix"
                ),
            }
        ],
    )
    return LaunchDescription(arguments + [node])

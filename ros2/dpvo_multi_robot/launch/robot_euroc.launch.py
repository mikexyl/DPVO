from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument("robot_id"),
        DeclareLaunchArgument("bag"),
        DeclareLaunchArgument("network"),
        DeclareLaunchArgument("config"),
        DeclareLaunchArgument("orb_vocab"),
        DeclareLaunchArgument("calib"),
        DeclareLaunchArgument("stride", default_value="2"),
        DeclareLaunchArgument("max_frames", default_value="0"),
        DeclareLaunchArgument("rerun_connect", default_value=""),
        DeclareLaunchArgument("rerun_recording_id", default_value=""),
        DeclareLaunchArgument("map_frame", default_value="map"),
        DeclareLaunchArgument("rerun_entity_prefix", default_value=""),
    ]
    robot_id = LaunchConfiguration("robot_id")
    nodes = [
        Node(
            package="dpvo_multi_robot",
            executable="dpvo_multi_robot_node",
            namespace=robot_id,
            name="dpvo",
            output="screen",
            parameters=[
                {
                    "robot_id": robot_id,
                    "network": LaunchConfiguration("network"),
                    "config": LaunchConfiguration("config"),
                    "orb_vocab": LaunchConfiguration("orb_vocab"),
                    "image_topic": "camera/image_raw",
                    "camera_info_topic": "camera/camera_info",
                    "pose_topic": "dpvo/pose",
                    "path_topic": "dpvo/path",
                    "frame_ack_topic": "dpvo/frame_ack",
                    "map_frame": LaunchConfiguration("map_frame"),
                    "rerun_connect": LaunchConfiguration("rerun_connect"),
                    "rerun_recording_id": LaunchConfiguration(
                        "rerun_recording_id"
                    ),
                    "rerun_entity_prefix": LaunchConfiguration(
                        "rerun_entity_prefix"
                    ),
                    "enable_dpvo_loop_closure": True,
                    "max_edge_age": 48,
                    "teaser_required": True,
                }
            ],
        ),
        Node(
            package="dpvo_multi_robot",
            executable="dpvo_euroc_player",
            namespace=robot_id,
            name="euroc_player",
            output="screen",
            parameters=[
                {
                    "bag": LaunchConfiguration("bag"),
                    "calib": LaunchConfiguration("calib"),
                    "image_topic": "camera/image_raw",
                    "camera_info_topic": "camera/camera_info",
                    "frame_ack_topic": "dpvo/frame_ack",
                    "done_topic": "dpvo/player_done",
                    "stride": ParameterValue(
                        LaunchConfiguration("stride"), value_type=int
                    ),
                    "max_frames": ParameterValue(
                        LaunchConfiguration("max_frames"), value_type=int
                    ),
                }
            ],
        ),
    ]
    return LaunchDescription(arguments + nodes)

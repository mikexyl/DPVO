"""Stage-one DPVO tracking for one robot stream in an S3E ROS2 bag."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument("bag"),
        DeclareLaunchArgument("calib"),
        DeclareLaunchArgument("source_topic"),
        DeclareLaunchArgument("robot_id"),
        DeclareLaunchArgument("network"),
        DeclareLaunchArgument("config"),
        DeclareLaunchArgument("orb_vocab"),
        DeclareLaunchArgument("tracking_artifact_output"),
        DeclareLaunchArgument("stride", default_value="2"),
        DeclareLaunchArgument("start_frame", default_value="0"),
        DeclareLaunchArgument("max_frames", default_value="0"),
        DeclareLaunchArgument("image_scale", default_value="0.5"),
        DeclareLaunchArgument("rectification_alpha", default_value="0.0"),
        DeclareLaunchArgument("random_seed", default_value="1234"),
        DeclareLaunchArgument("local_feature_backend", default_value="disk"),
        DeclareLaunchArgument("bow_threshold", default_value="0.01"),
        DeclareLaunchArgument("rerun_save", default_value=""),
        DeclareLaunchArgument("rerun_recording_id", default_value=""),
    ]
    tracker = Node(
        package="dpvo_multi_robot",
        executable="dpvo_multi_robot_node",
        namespace=LaunchConfiguration("robot_id"),
        name="dpvo",
        output="screen",
        parameters=[
            {
                "robot_id": LaunchConfiguration("robot_id"),
                "random_seed": ParameterValue(
                    LaunchConfiguration("random_seed"), value_type=int
                ),
                "network": LaunchConfiguration("network"),
                "config": LaunchConfiguration("config"),
                "orb_vocab": LaunchConfiguration("orb_vocab"),
                "image_topic": "camera/image_raw",
                "camera_info_topic": "camera/camera_info",
                "pose_topic": "dpvo/pose",
                "path_topic": "dpvo/path",
                "frame_ack_topic": "dpvo/frame_ack",
                "done_topic": "dpvo/player_done",
                "map_frame": "local_map",
                "image_scale": ParameterValue(
                    LaunchConfiguration("image_scale"), value_type=float
                ),
                "tracking_artifact_output": LaunchConfiguration(
                    "tracking_artifact_output"
                ),
                "exit_on_player_done": True,
                "enable_inter_robot_loop_closure": False,
                "enable_dpvo_loop_closure": True,
                "max_edge_age": 48,
                "retrieval_backend": "dbow2",
                "local_feature_backend": LaunchConfiguration(
                    "local_feature_backend"
                ),
                "bow_threshold": ParameterValue(
                    LaunchConfiguration("bow_threshold"), value_type=float
                ),
                "rerun_save": LaunchConfiguration("rerun_save"),
                "rerun_recording_id": LaunchConfiguration("rerun_recording_id"),
                "rerun_entity_prefix": "world/map",
            }
        ],
    )
    player = Node(
        package="dpvo_multi_robot",
        executable="dpvo_s3e_player",
        namespace=LaunchConfiguration("robot_id"),
        name="s3e_player",
        output="screen",
        parameters=[
            {
                "bag": LaunchConfiguration("bag"),
                "calib": LaunchConfiguration("calib"),
                "source_topic": LaunchConfiguration("source_topic"),
                "image_topic": "camera/image_raw",
                "camera_info_topic": "camera/camera_info",
                "frame_ack_topic": "dpvo/frame_ack",
                "done_topic": "dpvo/player_done",
                "exit_on_finish": True,
                "stride": ParameterValue(
                    LaunchConfiguration("stride"), value_type=int
                ),
                "start_frame": ParameterValue(
                    LaunchConfiguration("start_frame"), value_type=int
                ),
                "max_frames": ParameterValue(
                    LaunchConfiguration("max_frames"), value_type=int
                ),
                "rectification_alpha": ParameterValue(
                    LaunchConfiguration("rectification_alpha"), value_type=float
                ),
            }
        ],
    )
    stop_after_tracker = RegisterEventHandler(
        OnProcessExit(
            target_action=tracker,
            on_exit=[EmitEvent(event=Shutdown(reason="S3E tracking export complete"))],
        )
    )
    return LaunchDescription(arguments + [tracker, player, stop_after_tracker])

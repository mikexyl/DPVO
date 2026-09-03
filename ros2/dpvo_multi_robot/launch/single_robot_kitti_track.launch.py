"""Stage-one tracking for one KITTI partition with artifact export."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument("dataset_root"),
        DeclareLaunchArgument("sequence", default_value="00"),
        DeclareLaunchArgument("image_dir", default_value="image_0"),
        DeclareLaunchArgument("calibration_key", default_value="P0"),
        DeclareLaunchArgument("robot_id"),
        DeclareLaunchArgument("start_frame"),
        DeclareLaunchArgument("end_frame"),
        DeclareLaunchArgument("network"),
        DeclareLaunchArgument("config"),
        DeclareLaunchArgument("orb_vocab"),
        DeclareLaunchArgument("tracking_artifact_output"),
        DeclareLaunchArgument("stride", default_value="2"),
        DeclareLaunchArgument("image_scale", default_value="0.75"),
        DeclareLaunchArgument("max_frames", default_value="0"),
        DeclareLaunchArgument("random_seed", default_value="1234"),
        DeclareLaunchArgument("local_feature_backend", default_value="disk"),
        DeclareLaunchArgument("bow_threshold", default_value="0.03"),
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
                "rerun_recording_id": LaunchConfiguration(
                    "rerun_recording_id"
                ),
            }
        ],
    )
    player = Node(
        package="dpvo_multi_robot",
        executable="dpvo_kitti_player",
        namespace=LaunchConfiguration("robot_id"),
        name="kitti_player",
        output="screen",
        parameters=[
            {
                "sequence_dir": PathJoinSubstitution(
                    [
                        LaunchConfiguration("dataset_root"),
                        "sequences",
                        LaunchConfiguration("sequence"),
                    ]
                ),
                "image_dir": LaunchConfiguration("image_dir"),
                "calibration_key": LaunchConfiguration("calibration_key"),
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
                "end_frame": ParameterValue(
                    LaunchConfiguration("end_frame"), value_type=int
                ),
                "max_frames": ParameterValue(
                    LaunchConfiguration("max_frames"), value_type=int
                ),
            }
        ],
    )
    stop_after_player = RegisterEventHandler(
        OnProcessExit(
            target_action=player,
            on_exit=[
                EmitEvent(
                    event=Shutdown(reason="KITTI partition playback complete")
                )
            ],
        )
    )
    return LaunchDescription(arguments + [tracker, player, stop_after_player])

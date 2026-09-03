"""Stage-one three-robot DPVO tracking with no inter-robot verification."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


ROBOTS = ("robot0", "robot1", "robot2")


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument("dataset_root"),
        DeclareLaunchArgument("sequence0"),
        DeclareLaunchArgument("sequence1"),
        DeclareLaunchArgument("sequence2"),
        DeclareLaunchArgument("network"),
        DeclareLaunchArgument("config"),
        DeclareLaunchArgument("orb_vocab"),
        DeclareLaunchArgument("tracking_artifact_output"),
        DeclareLaunchArgument("stride", default_value="1"),
        DeclareLaunchArgument("image_scale", default_value="1.0"),
        DeclareLaunchArgument("max_frames", default_value="0"),
        DeclareLaunchArgument("random_seed", default_value="-1"),
        DeclareLaunchArgument("local_feature_backend", default_value="disk"),
        DeclareLaunchArgument("bow_threshold", default_value="0.01"),
        DeclareLaunchArgument("camera_crop_x", default_value="0"),
        DeclareLaunchArgument("camera_crop_y", default_value="0"),
        DeclareLaunchArgument("camera_fx"),
        DeclareLaunchArgument("camera_fy"),
        DeclareLaunchArgument("camera_cx"),
        DeclareLaunchArgument("camera_cy"),
        DeclareLaunchArgument("camera_k1", default_value="0.0"),
        DeclareLaunchArgument("camera_k2", default_value="0.0"),
        DeclareLaunchArgument("camera_p1", default_value="0.0"),
        DeclareLaunchArgument("camera_p2", default_value="0.0"),
        DeclareLaunchArgument("camera_k3", default_value="0.0"),
        DeclareLaunchArgument("rerun_connect", default_value=""),
        DeclareLaunchArgument(
            "rerun_recording_id", default_value="dpvo-offline-tracking"
        ),
    ]

    nodes = []
    for robot_index, robot_id in enumerate(ROBOTS):
        nodes.extend(
            [
                Node(
                    package="dpvo_multi_robot",
                    executable="dpvo_multi_robot_node",
                    namespace=robot_id,
                    name="dpvo",
                    output="screen",
                    parameters=[
                        {
                            "robot_id": robot_id,
                            "session_id": f"{robot_id}_offline_tracking",
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
                            "map_frame": f"{robot_id}_map",
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
                                LaunchConfiguration("bow_threshold"),
                                value_type=float,
                            ),
                            "rerun_connect": LaunchConfiguration("rerun_connect"),
                            "rerun_recording_id": LaunchConfiguration(
                                "rerun_recording_id"
                            ),
                            "rerun_entity_prefix": f"world/robots/{robot_id}/map",
                        }
                    ],
                ),
                Node(
                    package="dpvo_multi_robot",
                    executable="dpvo_tum_player",
                    namespace=robot_id,
                    name="tum_player",
                    output="screen",
                    parameters=[
                        {
                            "sequence_dir": PathJoinSubstitution(
                                [
                                    LaunchConfiguration("dataset_root"),
                                    LaunchConfiguration(f"sequence{robot_index}"),
                                ]
                            ),
                            "image_topic": "camera/image_raw",
                            "camera_info_topic": "camera/camera_info",
                            "frame_ack_topic": "dpvo/frame_ack",
                            "done_topic": "dpvo/player_done",
                            "exit_on_finish": True,
                            "stride": ParameterValue(
                                LaunchConfiguration("stride"), value_type=int
                            ),
                            "max_frames": ParameterValue(
                                LaunchConfiguration("max_frames"), value_type=int
                            ),
                            "crop_x": ParameterValue(
                                LaunchConfiguration("camera_crop_x"), value_type=int
                            ),
                            "crop_y": ParameterValue(
                                LaunchConfiguration("camera_crop_y"), value_type=int
                            ),
                            "fx": ParameterValue(
                                LaunchConfiguration("camera_fx"), value_type=float
                            ),
                            "fy": ParameterValue(
                                LaunchConfiguration("camera_fy"), value_type=float
                            ),
                            "cx": ParameterValue(
                                LaunchConfiguration("camera_cx"), value_type=float
                            ),
                            "cy": ParameterValue(
                                LaunchConfiguration("camera_cy"), value_type=float
                            ),
                            "k1": ParameterValue(
                                LaunchConfiguration("camera_k1"), value_type=float
                            ),
                            "k2": ParameterValue(
                                LaunchConfiguration("camera_k2"), value_type=float
                            ),
                            "p1": ParameterValue(
                                LaunchConfiguration("camera_p1"), value_type=float
                            ),
                            "p2": ParameterValue(
                                LaunchConfiguration("camera_p2"), value_type=float
                            ),
                            "k3": ParameterValue(
                                LaunchConfiguration("camera_k3"), value_type=float
                            ),
                        }
                    ],
                ),
            ]
        )

    return LaunchDescription(arguments + nodes)

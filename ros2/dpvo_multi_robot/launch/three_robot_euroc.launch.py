from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


ROBOTS = (
    ("robot0", "MH_01_easy.bag"),
    ("robot1", "MH_02_easy.bag"),
    ("robot2", "MH_03_medium.bag"),
)


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument("bag_root"),
        DeclareLaunchArgument("network"),
        DeclareLaunchArgument("config"),
        DeclareLaunchArgument("orb_vocab"),
        DeclareLaunchArgument("calib"),
        DeclareLaunchArgument("stride", default_value="2"),
        DeclareLaunchArgument("max_frames", default_value="0"),
        DeclareLaunchArgument("bow_threshold", default_value="0.04"),
        DeclareLaunchArgument("bow_repetitions", default_value="3"),
        DeclareLaunchArgument("bow_nms_radius", default_value="50"),
        DeclareLaunchArgument("teaser_noise_bound", default_value="0.10"),
        DeclareLaunchArgument("min_inliers", default_value="30"),
        DeclareLaunchArgument("min_inlier_ratio", default_value="0.20"),
        DeclareLaunchArgument("rerun_connect", default_value=""),
        DeclareLaunchArgument(
            "rerun_recording_id", default_value="dpvo-three-robot"
        ),
        DeclareLaunchArgument("pgo_output", default_value=""),
    ]

    nodes = [
        Node(
            package="dpvo_multi_robot",
            executable="dpvo_centralized_pgo",
            name="centralized_pgo",
            output="screen",
            parameters=[
                {
                    "robot_ids": [robot for robot, _ in ROBOTS],
                    "anchor_robot_id": "robot0",
                    "output_path": LaunchConfiguration("pgo_output"),
                    "rerun_connect": LaunchConfiguration("rerun_connect"),
                    "rerun_recording_id": LaunchConfiguration(
                        "rerun_recording_id"
                    ),
                }
            ],
        )
    ]

    for robot_id, bag_name in ROBOTS:
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
                            "network": LaunchConfiguration("network"),
                            "config": LaunchConfiguration("config"),
                            "orb_vocab": LaunchConfiguration("orb_vocab"),
                            "image_topic": "camera/image_raw",
                            "camera_info_topic": "camera/camera_info",
                            "pose_topic": "dpvo/pose",
                            "path_topic": "dpvo/path",
                            "frame_ack_topic": "dpvo/frame_ack",
                            "map_frame": f"{robot_id}_map",
                            "rerun_connect": LaunchConfiguration("rerun_connect"),
                            "rerun_recording_id": LaunchConfiguration(
                                "rerun_recording_id"
                            ),
                            "rerun_entity_prefix": (
                                f"world/robots/{robot_id}/map"
                            ),
                            "enable_dpvo_loop_closure": True,
                            "max_edge_age": 48,
                            "bow_threshold": ParameterValue(
                                LaunchConfiguration("bow_threshold"),
                                value_type=float,
                            ),
                            "bow_repetitions": ParameterValue(
                                LaunchConfiguration("bow_repetitions"),
                                value_type=int,
                            ),
                            "bow_nms_radius": ParameterValue(
                                LaunchConfiguration("bow_nms_radius"),
                                value_type=int,
                            ),
                            "teaser_noise_bound": ParameterValue(
                                LaunchConfiguration("teaser_noise_bound"),
                                value_type=float,
                            ),
                            "min_inliers": ParameterValue(
                                LaunchConfiguration("min_inliers"),
                                value_type=int,
                            ),
                            "min_inlier_ratio": ParameterValue(
                                LaunchConfiguration("min_inlier_ratio"),
                                value_type=float,
                            ),
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
                            "bag": PathJoinSubstitution(
                                [LaunchConfiguration("bag_root"), bag_name]
                            ),
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
        )

    return LaunchDescription(arguments + nodes)

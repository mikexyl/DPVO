from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


ROBOTS = ("robot0", "robot1")
DEFAULT_SEQUENCES = (
    "rgbd_dataset_freiburg1_desk",
    "rgbd_dataset_freiburg1_desk2",
)


def generate_launch_description():
    arguments = [
        DeclareLaunchArgument("dataset_root"),
        *[
            DeclareLaunchArgument(
                f"sequence{robot_index}",
                default_value=DEFAULT_SEQUENCES[robot_index],
            )
            for robot_index in range(len(ROBOTS))
        ],
        DeclareLaunchArgument("network"),
        DeclareLaunchArgument("config"),
        DeclareLaunchArgument("orb_vocab"),
        DeclareLaunchArgument("stride", default_value="1"),
        DeclareLaunchArgument("image_scale", default_value="1.0"),
        DeclareLaunchArgument("max_frames", default_value="0"),
        DeclareLaunchArgument("random_seed", default_value="-1"),
        DeclareLaunchArgument("retrieval_backend", default_value="megaloc"),
        DeclareLaunchArgument("megaloc_repo", default_value="gmberton/MegaLoc"),
        DeclareLaunchArgument("megaloc_model_id", default_value="gmberton/MegaLoc"),
        DeclareLaunchArgument("megaloc_threshold", default_value="0.20"),
        DeclareLaunchArgument("local_feature_backend", default_value="xfeat"),
        DeclareLaunchArgument(
            "xfeat_repo", default_value="verlab/accelerated_features"
        ),
        DeclareLaunchArgument("xfeat_top_k", default_value="2048"),
        DeclareLaunchArgument("xfeat_detection_threshold", default_value="0.05"),
        DeclareLaunchArgument("lightglue_min_confidence", default_value="0.10"),
        DeclareLaunchArgument("bow_threshold", default_value="0.03"),
        DeclareLaunchArgument("bow_repetitions", default_value="2"),
        DeclareLaunchArgument("bow_nms_radius", default_value="15"),
        DeclareLaunchArgument("bow_backfill", default_value="true"),
        DeclareLaunchArgument("teaser_noise_bound", default_value="0.10"),
        DeclareLaunchArgument("min_inliers", default_value="25"),
        DeclareLaunchArgument("min_inlier_ratio", default_value="0.15"),
        DeclareLaunchArgument("max_depth", default_value="20.0"),
        DeclareLaunchArgument(
            "loop_diagnostics_dir", default_value="/tmp/dpvo-loop-diagnostics"
        ),
        DeclareLaunchArgument("loop_diagnostics_period", default_value="5.0"),
        DeclareLaunchArgument("camera_crop_x", default_value="16"),
        DeclareLaunchArgument("camera_crop_y", default_value="8"),
        DeclareLaunchArgument("camera_fx", default_value="517.3"),
        DeclareLaunchArgument("camera_fy", default_value="516.5"),
        DeclareLaunchArgument("camera_cx", default_value="318.6"),
        DeclareLaunchArgument("camera_cy", default_value="255.3"),
        DeclareLaunchArgument("camera_k1", default_value="0.2624"),
        DeclareLaunchArgument("camera_k2", default_value="-0.9531"),
        DeclareLaunchArgument("camera_p1", default_value="-0.0054"),
        DeclareLaunchArgument("camera_p2", default_value="0.0026"),
        DeclareLaunchArgument("camera_k3", default_value="1.1633"),
        DeclareLaunchArgument("rerun_connect", default_value=""),
        DeclareLaunchArgument(
            "rerun_recording_id", default_value="dpvo-two-robot-tum"
        ),
        DeclareLaunchArgument("pgo_output", default_value=""),
        DeclareLaunchArgument("pose_graph_output", default_value=""),
        DeclareLaunchArgument("pose_graph_export_period", default_value="0.0"),
        DeclareLaunchArgument("pose_graph_odometry_weight", default_value="100.0"),
        DeclareLaunchArgument("enable_cbs", default_value="true"),
        DeclareLaunchArgument(
            "cbs_executable", default_value="cbs_dpvo_sim3_offline"
        ),
        DeclareLaunchArgument("cbs_output_dir", default_value="/tmp/dpvo-cbs"),
        DeclareLaunchArgument("cbs_iterations", default_value="200"),
        DeclareLaunchArgument("cbs_stage_mode", default_value="alternating"),
        DeclareLaunchArgument("cbs_target_hellinger", default_value="0.1"),
        DeclareLaunchArgument("cbs_d_reset", default_value="0.6"),
        DeclareLaunchArgument("cbs_settle_seconds", default_value="30.0"),
        DeclareLaunchArgument("cbs_run_centralized_baseline", default_value="true"),
        DeclareLaunchArgument(
            "cbs_run_explicit_anchor_centralized_baseline", default_value="true"
        ),
    ]

    nodes = [
        Node(
            package="dpvo_multi_robot",
            executable="dpvo_centralized_pgo",
            name="centralized_pgo",
            output="screen",
            parameters=[
                {
                    "robot_ids": list(ROBOTS),
                    "anchor_robot_id": "robot0",
                    "output_path": LaunchConfiguration("pgo_output"),
                    "pose_graph_output": LaunchConfiguration(
                        "pose_graph_output"
                    ),
                    "pose_graph_export_period": ParameterValue(
                        LaunchConfiguration("pose_graph_export_period"),
                        value_type=float,
                    ),
                    "pose_graph_odometry_weight": ParameterValue(
                        LaunchConfiguration("pose_graph_odometry_weight"),
                        value_type=float,
                    ),
                    "rerun_connect": LaunchConfiguration("rerun_connect"),
                    "rerun_recording_id": LaunchConfiguration(
                        "rerun_recording_id"
                    ),
                }
            ],
        ),
        Node(
            package="dpvo_multi_robot",
            executable="dpvo_cbs_pgo",
            name="cbs_pgo",
            output="screen",
            condition=IfCondition(LaunchConfiguration("enable_cbs")),
            parameters=[
                {
                    "robot_ids": list(ROBOTS),
                    "anchor_robot_id": "robot0",
                    "output_dir": LaunchConfiguration("cbs_output_dir"),
                    "cbs_executable": LaunchConfiguration("cbs_executable"),
                    "iterations": ParameterValue(
                        LaunchConfiguration("cbs_iterations"), value_type=int
                    ),
                    "stage_mode": LaunchConfiguration("cbs_stage_mode"),
                    "target_hellinger": ParameterValue(
                        LaunchConfiguration("cbs_target_hellinger"),
                        value_type=float,
                    ),
                    "d_reset": ParameterValue(
                        LaunchConfiguration("cbs_d_reset"), value_type=float
                    ),
                    "settle_seconds": ParameterValue(
                        LaunchConfiguration("cbs_settle_seconds"),
                        value_type=float,
                    ),
                    "run_centralized_baseline": ParameterValue(
                        LaunchConfiguration("cbs_run_centralized_baseline"),
                        value_type=bool,
                    ),
                    "run_explicit_anchor_centralized_baseline": ParameterValue(
                        LaunchConfiguration(
                            "cbs_run_explicit_anchor_centralized_baseline"
                        ),
                        value_type=bool,
                    ),
                    "pose_graph_odometry_weight": ParameterValue(
                        LaunchConfiguration("pose_graph_odometry_weight"),
                        value_type=float,
                    ),
                    "rerun_connect": LaunchConfiguration("rerun_connect"),
                    "rerun_recording_id": LaunchConfiguration(
                        "rerun_recording_id"
                    ),
                }
            ],
        ),
    ]

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
                            "map_frame": f"{robot_id}_map",
                            "image_scale": ParameterValue(
                                LaunchConfiguration("image_scale"), value_type=float
                            ),
                            "rerun_connect": LaunchConfiguration("rerun_connect"),
                            "rerun_recording_id": LaunchConfiguration(
                                "rerun_recording_id"
                            ),
                            "rerun_entity_prefix": (
                                f"world/robots/{robot_id}/map"
                            ),
                            "enable_dpvo_loop_closure": True,
                            "max_edge_age": 48,
                            "retrieval_backend": LaunchConfiguration(
                                "retrieval_backend"
                            ),
                            "megaloc_repo": LaunchConfiguration("megaloc_repo"),
                            "megaloc_model_id": LaunchConfiguration(
                                "megaloc_model_id"
                            ),
                            "megaloc_threshold": ParameterValue(
                                LaunchConfiguration("megaloc_threshold"),
                                value_type=float,
                            ),
                            "local_feature_backend": LaunchConfiguration(
                                "local_feature_backend"
                            ),
                            "xfeat_repo": LaunchConfiguration("xfeat_repo"),
                            "xfeat_top_k": ParameterValue(
                                LaunchConfiguration("xfeat_top_k"),
                                value_type=int,
                            ),
                            "xfeat_detection_threshold": ParameterValue(
                                LaunchConfiguration("xfeat_detection_threshold"),
                                value_type=float,
                            ),
                            "lightglue_min_confidence": ParameterValue(
                                LaunchConfiguration("lightglue_min_confidence"),
                                value_type=float,
                            ),
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
                            "bow_backfill": ParameterValue(
                                LaunchConfiguration("bow_backfill"),
                                value_type=bool,
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
                            "max_depth": ParameterValue(
                                LaunchConfiguration("max_depth"),
                                value_type=float,
                            ),
                            "loop_diagnostics_output": PathJoinSubstitution(
                                [
                                    LaunchConfiguration("loop_diagnostics_dir"),
                                    f"{robot_id}.json",
                                ]
                            ),
                            "loop_diagnostics_period": ParameterValue(
                                LaunchConfiguration("loop_diagnostics_period"),
                                value_type=float,
                            ),
                            "teaser_required": True,
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

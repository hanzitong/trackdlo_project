"""
trackdlo_node.launch.py

ROS2ノード trackdlo_node を起動するlaunchファイル。
RealSense カメラと合わせて使うことを想定している。

使い方:
    ros2 launch trackdlo trackdlo_node.launch.py
    ros2 launch trackdlo trackdlo_node.launch.py hsv_lower:="90 90 30" hsv_upper:="130 255 255"
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():

    # ---- launch 引数 ----
    # カメラトピック / フレーム
    declare_camera_info  = DeclareLaunchArgument(
        "camera_info_topic",
        default_value="/camera/aligned_depth_to_color/camera_info")
    declare_rgb          = DeclareLaunchArgument(
        "rgb_topic",
        default_value="/camera/color/image_raw")
    declare_depth        = DeclareLaunchArgument(
        "depth_topic",
        default_value="/camera/aligned_depth_to_color/image_raw")
    declare_frame        = DeclareLaunchArgument(
        "result_frame_id",
        default_value="camera_color_optical_frame")

    # HSV閾値 (スペース区切り文字列)
    declare_hsv_upper = DeclareLaunchArgument(
        "hsv_upper",
        default_value="130 255 255",
        description="HSV上限 (例: '130 255 255')")
    declare_hsv_lower = DeclareLaunchArgument(
        "hsv_lower",
        default_value="90 90 30",
        description="HSV下限 (例: '90 90 30')")

    # マルチカラー DLO モード
    declare_multi_color = DeclareLaunchArgument(
        "multi_color_dlo",
        default_value="false")

    # ---- ノード定義 ----
    trackdlo_node = Node(
        package="trackdlo",
        executable="trackdlo_node",
        name="trackdlo_node",
        output="screen",
        parameters=[{
            # トピック / フレーム
            "camera_info_topic":          LaunchConfiguration("camera_info_topic"),
            "rgb_topic":                  LaunchConfiguration("rgb_topic"),
            "depth_topic":                LaunchConfiguration("depth_topic"),
            "result_frame_id":            LaunchConfiguration("result_frame_id"),

            # HSV閾値
            "hsv_threshold_upper_limit":  LaunchConfiguration("hsv_upper"),
            "hsv_threshold_lower_limit":  LaunchConfiguration("hsv_lower"),

            # アルゴリズムパラメータ
            "beta":                       0.35,
            "lambda":                     50000.0,
            "alpha":                      3.0,
            "mu":                         0.1,
            "max_iter":                   50,
            "tol":                        0.0002,
            "k_vis":                      50.0,
            "visibility_threshold":       0.008,
            "beta_pre_proc":              3.0,
            "lambda_pre_proc":            1.0,
            "lle_weight":                 10.0,

            # 前処理パラメータ
            "d_vis":                      0.06,
            "dlo_pixel_width":            40,
            "downsample_leaf_size":       0.008,
            "multi_color_dlo":            LaunchConfiguration("multi_color_dlo"),
        }]
    )

    return LaunchDescription([
        declare_camera_info,
        declare_rgb,
        declare_depth,
        declare_frame,
        declare_hsv_upper,
        declare_hsv_lower,
        declare_multi_color,
        trackdlo_node,
    ])

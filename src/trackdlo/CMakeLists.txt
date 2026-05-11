cmake_minimum_required(VERSION 3.14)
project(trackdlo)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-O3)

# ---- ROS2 依存 ----
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)
find_package(pcl_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(message_filters REQUIRED)

# ---- 外部ライブラリ ----
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV REQUIRED)
find_package(PCL 1.8 REQUIRED COMPONENTS common io filters)

# ---- 同ワークスペース内の非ROS2ライブラリ ----
# colcon は package.xml の <depend> を見てビルド順を制御する。
# pure_trackdlo と preprocessing が先にビルド・インストールされた後、
# CMAKE_PREFIX_PATH 経由でここで find できる。
find_package(pure_trackdlo REQUIRED)
find_package(preprocessing REQUIRED)

# ---- ROS2 ノード (trackdlo_node.cpp が ROS2 対応版に書き換え後に有効にする) ----
# add_executable(trackdlo_node src/trackdlo_node.cpp)
# target_link_libraries(trackdlo_node
#   pure_trackdlo   # cmake export target (Eigen3 も推移的に渡る)
#   preprocessing   # cmake export target
#   ${OpenCV_LIBS}
#   ${PCL_LIBRARIES}
# )
# ament_target_dependencies(trackdlo_node
#   rclcpp
#   sensor_msgs
#   std_msgs
#   visualization_msgs
#   cv_bridge
#   image_transport
#   pcl_ros
#   pcl_conversions
#   message_filters
# )
# install(TARGETS trackdlo_node DESTINATION lib/${PROJECT_NAME})

# ---- launch / config / rviz / scripts のインストール ----
install(DIRECTORY launch config rviz scripts
    DESTINATION share/${PROJECT_NAME}
)

ament_package()

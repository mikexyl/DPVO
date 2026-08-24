cmake_minimum_required(VERSION 3.8)
project(dpvo_multi_robot_interfaces)

find_package(ament_cmake REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/BowVector.msg"
  "msg/InterRobotLoopClosure.msg"
  "msg/KeyframeData.msg"
  "msg/RobotMapTransform.msg"
  "srv/GetKeyframe.srv"
  DEPENDENCIES geometry_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()

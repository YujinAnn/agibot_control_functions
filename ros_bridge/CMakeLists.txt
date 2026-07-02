cmake_minimum_required(VERSION 3.16)
project(ros_bridge)

if(NOT CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 17)
endif()
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(aimdk_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

add_executable(robot_ros_bridge robot_ros_bridge.cpp)
ament_target_dependencies(robot_ros_bridge rclcpp aimdk_msgs sensor_msgs)

install(TARGETS robot_ros_bridge DESTINATION lib/${PROJECT_NAME})
ament_package()

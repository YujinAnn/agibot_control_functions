cmake_minimum_required(VERSION 3.16)
project(mujoco_x2_cpp)

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
find_package(yaml-cpp REQUIRED)
find_package(OpenGL REQUIRED)

# ---- MuJoCo location (contains include/ and libmujoco.so*) --------------------
# Resolve in priority order so this builds on any machine without editing:
#   1) -DMUJOCO_DIR=...            (explicit)
#   2) $MUJOCO_DIR                 (environment)
#   3) the mujoco/ dir of the active conda env ($CONDA_PREFIX)
# Override at configure time, e.g.:
#   colcon build --cmake-args -DMUJOCO_DIR=/path/to/site-packages/mujoco
if(NOT MUJOCO_DIR)
  if(DEFINED ENV{MUJOCO_DIR})
    set(MUJOCO_DIR "$ENV{MUJOCO_DIR}")
  elseif(DEFINED ENV{CONDA_PREFIX})
    file(GLOB _mj_cands "$ENV{CONDA_PREFIX}/lib/python*/site-packages/mujoco")
    if(_mj_cands)
      list(GET _mj_cands 0 MUJOCO_DIR)
    endif()
  endif()
endif()
if(NOT MUJOCO_DIR OR NOT EXISTS "${MUJOCO_DIR}/include/mujoco/mujoco.h")
  message(FATAL_ERROR
    "MuJoCo not found. Pass -DMUJOCO_DIR=/path/to/mujoco (the dir with "
    "include/mujoco/mujoco.h and libmujoco.so), or set the MUJOCO_DIR env var, "
    "or activate the conda env that has the mujoco pip package.")
endif()
message(STATUS "Using MuJoCo at: ${MUJOCO_DIR}")
file(GLOB MUJOCO_LIB "${MUJOCO_DIR}/libmujoco.so*")

find_library(GLFW_LIB NAMES glfw glfw3
    HINTS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib "$ENV{CONDA_PREFIX}/lib")

# ---- target ------------------------------------------------------------------
# No repo-root path is baked in: run_mujoco_x2.cpp locates its data files
# (robots/x2/...) at runtime, so the built binary is relocatable.
add_executable(run_mujoco_x2 run_mujoco_x2.cpp)
target_include_directories(run_mujoco_x2 PRIVATE
  "${MUJOCO_DIR}/include"
  ${YAML_CPP_INCLUDE_DIR})
target_link_libraries(run_mujoco_x2
  ${MUJOCO_LIB}
  ${GLFW_LIB}
  OpenGL::GL
  ${YAML_CPP_LIBRARIES})
ament_target_dependencies(run_mujoco_x2 rclcpp aimdk_msgs sensor_msgs)
set_target_properties(run_mujoco_x2 PROPERTIES
  BUILD_RPATH "${MUJOCO_DIR}"
  INSTALL_RPATH "${MUJOCO_DIR}")

install(TARGETS run_mujoco_x2
        DESTINATION lib/${PROJECT_NAME})
ament_package()

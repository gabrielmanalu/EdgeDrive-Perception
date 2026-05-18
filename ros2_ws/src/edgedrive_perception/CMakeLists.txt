cmake_minimum_required(VERSION 3.18)
project(edgedrive_perception CUDA CXX)

# ── C++ / CUDA standard ───────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_ARCHITECTURES 87)  # Jetson Orin Nano Super SM87

# ── ROS2 dependencies ─────────────────────────────────────────────────────────
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(image_transport REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(message_filters REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# ── System dependencies ───────────────────────────────────────────────────────
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)

# TensorRT
set(TENSORRT_INCLUDE_DIR /usr/include/aarch64-linux-gnu)
set(TENSORRT_LIBRARY     /usr/lib/aarch64-linux-gnu/libnvinfer.so)
set(TENSORRT_PLUGIN_LIB  /usr/lib/aarch64-linux-gnu/libnvinfer_plugin.so)

# ── Shared sources from deployment/ ───────────────────────────────────────────
# Repo is mounted at /workspace inside the dusty-nv container
# On host: ~/EdgeDrive-Perception/deployment/
# In container: /workspace/deployment/
if(EXISTS /workspace/deployment/src/trt_engine.cpp)
    # Running inside dusty-nv container
    set(DEPLOYMENT_SRC /workspace/deployment/src)
    set(DEPLOYMENT_INC /workspace/deployment/include)
else()
    # Running on host (native colcon build)
    set(DEPLOYMENT_SRC ${CMAKE_SOURCE_DIR}/../../deployment/src)
    set(DEPLOYMENT_INC ${CMAKE_SOURCE_DIR}/../../deployment/include)
endif()

set(SHARED_SOURCES
    ${DEPLOYMENT_SRC}/trt_engine.cpp
    ${DEPLOYMENT_SRC}/yolo26_decoder.cpp
    ${DEPLOYMENT_SRC}/profiler.cpp
    ${DEPLOYMENT_SRC}/bev_visualizer.cpp
)

# ── camera_node ───────────────────────────────────────────────────────────────
add_executable(camera_node
    src/camera_node.cpp
    ${SHARED_SOURCES}
)

target_include_directories(camera_node PRIVATE
    include
    ${DEPLOYMENT_INC}
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(camera_node
    ${TENSORRT_LIBRARY}
    ${TENSORRT_PLUGIN_LIB}
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBS}
    pthread
    -Wl,--allow-shlib-undefined
)

ament_target_dependencies(camera_node
    rclcpp
    sensor_msgs
    std_msgs
    vision_msgs
    visualization_msgs
    image_transport
    cv_bridge
    tf2_ros
)

# ── fusion_node (placeholder — wired in June with PointPillars) ──────────────
# add_executable(fusion_node
#     src/fusion_node.cpp
#     ${SHARED_SOURCES}
# )
# ament_target_dependencies(fusion_node ...)

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS camera_node
    DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY
    launch
    config
    DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY include/
    DESTINATION include
)

ament_package()
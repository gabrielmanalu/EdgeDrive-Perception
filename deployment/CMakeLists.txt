cmake_minimum_required(VERSION 3.18)
project(EdgeDrivePerception LANGUAGES CXX CUDA)

# ── Standards ─────────────────────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# ── CUDA ──────────────────────────────────────────────────────────────────────
find_package(CUDA REQUIRED)
# Jetson Orin Nano Super = SM 8.7
set(CMAKE_CUDA_ARCHITECTURES 87)
include_directories(${CUDA_INCLUDE_DIRS})

# ── TensorRT ──────────────────────────────────────────────────────────────────
find_path(TENSORRT_INCLUDE_DIR
  NAMES NvInfer.h
  HINTS /usr/include/aarch64-linux-gnu /usr/include
)
find_library(TENSORRT_LIBRARY      NAMES nvinfer
  HINTS /usr/lib/aarch64-linux-gnu /usr/lib)
find_library(TENSORRT_PLUGIN_LIB   NAMES nvinfer_plugin
  HINTS /usr/lib/aarch64-linux-gnu /usr/lib)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_LIBRARY)
  message(FATAL_ERROR "TensorRT not found")
endif()
message(STATUS "TensorRT: ${TENSORRT_LIBRARY}")
include_directories(${TENSORRT_INCLUDE_DIR})

# ── OpenCV ────────────────────────────────────────────────────────────────────
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS "OpenCV: ${OpenCV_VERSION}")

# ── Project headers ───────────────────────────────────────────────────────────
include_directories(${CMAKE_SOURCE_DIR}/include)

# ── Core sources only (empty stubs excluded) ──────────────────────────────────
set(SOURCES
  src/trt_engine.cpp
  src/yolo26_decoder.cpp
  src/profiler.cpp
  src/camera_capture.cpp
  src/main.cpp
)

# ── Executable ────────────────────────────────────────────────────────────────
add_executable(edgedrive ${SOURCES})

target_link_libraries(edgedrive
  ${TENSORRT_LIBRARY}
  ${TENSORRT_PLUGIN_LIB}
  ${CUDA_LIBRARIES}
  /usr/local/cuda-12.6/targets/aarch64-linux/lib/libcudart.so
  ${OpenCV_LIBS}
  pthread
)

# ── Summary ───────────────────────────────────────────────────────────────────
message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "CUDA arch  : ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Sources    : ${SOURCES}")
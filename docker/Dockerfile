# ============================================================================
# EdgeDrive Perception — Jetson Runtime Container
# ============================================================================
#
# Multi-stage build:
#   Stage 1 (builder): compile C++ pipeline inside container
#   Stage 2 (runtime): minimal image with binary + dependencies only
#
# Base: nvcr.io/nvidia/l4t-tensorrt:r36.3.0
#   Compatible with JetPack R36.3.0 - R36.4.x (our target: R36.4.7)
#   Includes: CUDA 12.6, TensorRT 10.x, cuDNN
#
# Build on Jetson (ARM64 only — TRT engines are hardware-specific):
#   docker build -t edgedrive:latest -f docker/Dockerfile .
#
# Run:
#   docker compose -f docker/docker-compose.yml up camera
# ============================================================================

# ── Stage 1: Builder ─────────────────────────────────────────────────────────

FROM nvcr.io/nvidia/l4t-tensorrt:r36.3.0-devel AS builder

LABEL stage=builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    g++ \
    ninja-build \
    pkg-config \
    # OpenCV 4.8.0 (L4T build with CUDA + GStreamer support)
    libopencv-dev \
    # GStreamer (for NVDEC video decode and CSI camera)
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy source
COPY deployment/CMakeLists.txt .
COPY deployment/src/ src/
COPY deployment/include/ include/

# Build
RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -G Ninja \
    && cmake --build build --parallel 4

# ── Stage 2: Runtime ─────────────────────────────────────────────────────────

FROM nvcr.io/nvidia/l4t-tensorrt:r36.3.0-runtime AS runtime

LABEL maintainer="EdgeDrive Perception"
LABEL description="YOLO26n TRT INT8 inference — Jetson Orin Nano Super"
LABEL version="1.0"

# Runtime dependencies only (no dev headers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    # OpenCV runtime
    libopencv-core4.8 \
    libopencv-imgproc4.8 \
    libopencv-highgui4.8 \
    libopencv-videoio4.8 \
    libopencv-imgcodecs4.8 \
    # GStreamer runtime + NVDEC
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    # V4L2 for USB camera
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled binary from builder
COPY --from=builder /build/build/edgedrive /usr/local/bin/edgedrive

# Working directory — engines and videos mounted here at runtime
WORKDIR /workspace

# Default command: show usage
ENTRYPOINT ["/usr/local/bin/edgedrive"]
CMD ["--help"]

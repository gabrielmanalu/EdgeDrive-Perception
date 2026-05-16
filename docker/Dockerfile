# ============================================================================
# EdgeDrive Perception — Jetson Runtime Container
# ============================================================================
#
# Multi-stage build:
#   Stage 1 (builder): compile C++ pipeline inside container
#   Stage 2 (runtime): minimal image with binary + dependencies only
#
# Base: nvcr.io/nvidia/l4t-jetpack:r36.4.0
#   Compatible with JetPack R36.4.x (our target: R36.4.7)
#   Includes: CUDA 12.6, TensorRT 10.x, cuDNN, OpenCV 4.8.0
#
# Linker note:
#   libnvdla_compiler.so and libnvcudla.so are Jetson BSP libs that exist
#   on the host but not inside the container image. They are injected at
#   runtime by --runtime nvidia. We use -Wl,--allow-shlib-undefined to
#   skip them at link time — safe because we don't use DLA directly.
#
# Build on Jetson only (ARM64 — TRT engines are device-specific):
#   docker build -t edgedrive:latest -f docker/Dockerfile .
# ============================================================================

# ── Stage 1: Builder ─────────────────────────────────────────────────────────

FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0 AS builder

LABEL stage=builder

# Build tools + GStreamer dev headers
# OpenCV 4.8, CUDA 12.6, TRT already in l4t-jetpack
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake \
    ninja-build \
    g++ \
    pkg-config \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY deployment/CMakeLists.txt .
COPY deployment/src/ src/
COPY deployment/include/ include/

# -Wl,--allow-shlib-undefined: skip missing Jetson BSP libs at link time
# (libnvdla_compiler.so, libnvcudla.so — injected at runtime by --runtime nvidia)
RUN cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -G Ninja \
    && cmake --build build --parallel 4

# ── Stage 2: Runtime ─────────────────────────────────────────────────────────

FROM nvcr.io/nvidia/l4t-jetpack:r36.4.0 AS runtime

LABEL maintainer="EdgeDrive Perception"
LABEL description="YOLO26n TRT INT8 — Jetson Orin Nano Super 203 FPS / 8W"
LABEL version="1.0"

# GStreamer runtime plugins for NVDEC + USB camera
# OpenCV, CUDA, TRT already in l4t-jetpack
RUN apt-get update && apt-get install -y --no-install-recommends \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-libav \
    v4l-utils \
    && rm -rf /var/lib/apt/lists/*

# Copy compiled binary from builder stage
COPY --from=builder /build/build/edgedrive /usr/local/bin/edgedrive

# Engines and videos mounted at runtime — not baked into image
WORKDIR /workspace

ENTRYPOINT ["/usr/local/bin/edgedrive"]
CMD ["--help"]
#pragma once
#include <cuda_runtime.h>

/**
 * preprocessor.cuh — CUDA letterbox preprocessing kernel
 * ========================================================
 * GPU-side equivalent of the CPU preprocessing in trt_engine.cpp.
 *
 * CPU pipeline (current):
 *   cv::resize → cv::cvtColor → pixel loop → write to pinned buffer
 *   Time: ~1.1-1.7ms (depends on input resolution)
 *
 * CUDA pipeline (this file):
 *   launchLetterboxKernel() → one GPU kernel pass
 *   Time: ~0.2-0.3ms (estimated)
 *
 * Why it's not the default:
 *   At 200 FPS, 1.1ms CPU preprocessing is not the bottleneck.
 *   TRT inference (3.7ms) is the ceiling.
 *   CUDA kernel is useful for:
 *     - Multi-camera pipelines (6× cameras simultaneously)
 *     - When TRT inference < 1ms (smaller/faster future model)
 *     - Demonstrating CUDA kernel skills
 *
 * Usage:
 *   // Allocate input buffer on device (or use UMA)
 *   float* d_output;  // [3 × 640 × 640] CHW float32
 *   uint8_t* d_input; // [H × W × 3] HWC uint8 BGR
 *
 *   launchLetterboxKernel(
 *       d_input, d_output,
 *       src_w, src_h,
 *       640, 640,
 *       stream);
 */

/**
 * Letterbox + BGR→RGB + HWC→CHW + normalize [0,255]→[0,1]
 * All in one kernel pass.
 *
 * @param src       Input image (HWC, uint8, BGR) on device
 * @param dst       Output tensor (CHW, float32, RGB) on device
 * @param src_w     Source width
 * @param src_h     Source height
 * @param dst_w     Target width  (640)
 * @param dst_h     Target height (640)
 * @param pad_val   Letterbox fill value (default 114)
 * @param stream    CUDA stream
 */
void launchLetterboxKernel(
    const uint8_t* src,
    float*         dst,
    int src_w, int src_h,
    int dst_w, int dst_h,
    uint8_t pad_val = 114,
    cudaStream_t stream = 0);

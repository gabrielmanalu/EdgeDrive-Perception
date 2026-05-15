/**
 * preprocessor.cu — CUDA letterbox preprocessing kernel
 *
 * One kernel pass replaces: cv::resize + cv::cvtColor + pixel loop
 * Each thread handles one output pixel.
 */

#include "preprocessor.cuh"
#include <cuda_runtime.h>

// ── Letterbox kernel ──────────────────────────────────────────────────────────

__global__ void letterboxKernel(
    const uint8_t* __restrict__ src,
    float*         __restrict__ dst,
    int src_w, int src_h,
    int dst_w, int dst_h,
    float scale,       // src → dst scale factor
    int   pad_top,     // vertical padding (pixels)
    int   pad_left,    // horizontal padding (pixels)
    float inv255,      // 1.0f / 255.0f  (precomputed)
    uint8_t pad_val)
{
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x >= dst_w || dst_y >= dst_h) return;

    int src_x = static_cast<int>((dst_x - pad_left) / scale);
    int src_y = static_cast<int>((dst_y - pad_top)  / scale);

    float r, g, b;

    if (src_x < 0 || src_x >= src_w || src_y < 0 || src_y >= src_h) {
        // Padding region — fill value
        r = g = b = pad_val * inv255;
    } else {
        // BGR → RGB (swap B and R channels)
        int idx = (src_y * src_w + src_x) * 3;
        b = src[idx + 0] * inv255;  // B
        g = src[idx + 1] * inv255;  // G
        r = src[idx + 2] * inv255;  // R
    }

    // Write CHW layout: [C, H, W]
    int pixel = dst_y * dst_w + dst_x;
    dst[0 * dst_h * dst_w + pixel] = r;  // R channel
    dst[1 * dst_h * dst_w + pixel] = g;  // G channel
    dst[2 * dst_h * dst_w + pixel] = b;  // B channel
}

// ── Launch function ───────────────────────────────────────────────────────────

void launchLetterboxKernel(
    const uint8_t* src,
    float*         dst,
    int src_w, int src_h,
    int dst_w, int dst_h,
    uint8_t pad_val,
    cudaStream_t stream)
{
    // Compute letterbox scale and padding
    float scale = fminf(
        static_cast<float>(dst_w) / src_w,
        static_cast<float>(dst_h) / src_h);

    int scaled_w = static_cast<int>(src_w * scale);
    int scaled_h = static_cast<int>(src_h * scale);
    int pad_left = (dst_w - scaled_w) / 2;
    int pad_top  = (dst_h - scaled_h) / 2;

    // 16×16 thread blocks — 256 threads, good for 2D image workloads
    dim3 block(16, 16);
    dim3 grid(
        (dst_w + block.x - 1) / block.x,
        (dst_h + block.y - 1) / block.y);

    letterboxKernel<<<grid, block, 0, stream>>>(
        src, dst,
        src_w, src_h,
        dst_w, dst_h,
        scale,
        pad_top, pad_left,
        1.0f / 255.0f,
        pad_val);
}

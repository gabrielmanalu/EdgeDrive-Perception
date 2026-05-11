#pragma once

#include <string>
#include <vector>
#include <memory>
#include <iostream>

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

// ── TensorRT logger ───────────────────────────────────────────────────────────

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // Suppress INFO and VERBOSE messages — only show warnings/errors
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TRT] " << msg << std::endl;
        }
    }
};

// ── TRTEngine ─────────────────────────────────────────────────────────────────

class TRTEngine {
public:
    // Constructor / Destructor
    explicit TRTEngine(const std::string& engine_path);
    ~TRTEngine();

    // Disable copy
    TRTEngine(const TRTEngine&)            = delete;
    TRTEngine& operator=(const TRTEngine&) = delete;

    // Core inference
    // Runs full pipeline: preprocess → inference → returns raw output buffer
    // frame: BGR image from OpenCV (any size, will be resized to input_w x input_h)
    // Returns pointer to output buffer (device-mapped, valid until next infer())
    float* infer(const cv::Mat& frame);

    // Accessors
    int    inputWidth()   const { return input_w_; }
    int    inputHeight()  const { return input_h_; }
    int    outputSize()   const { return output_size_; }
    float  lastInferMs()  const { return last_infer_ms_; }

private:
    // Engine loading
    void loadEngine(const std::string& engine_path);
    void allocateBuffers();

    // Preprocessing — resize + normalize → input buffer
    // Writes directly into pinned input_host_ buffer (zero-copy on Jetson UMA)
    void preprocess(const cv::Mat& frame);

    // TensorRT objects
    Logger                                      logger_;
    std::unique_ptr<nvinfer1::IRuntime>         runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine>      engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    // CUDA stream for async execution
    cudaStream_t stream_;

    // Pinned (page-locked) memory — zero-copy on Jetson unified memory
    // CPU writes → GPU reads the same physical memory, no cudaMemcpy needed
    float* input_host_;    // CPU-writable pointer
    float* input_device_;  // GPU-readable pointer (same physical memory)
    float* output_host_;   // CPU-readable pointer
    float* output_device_; // GPU-writable pointer (same physical memory)

    // Buffer sizes
    int input_w_    = 640;
    int input_h_    = 640;
    int input_size_ = 0;   // total floats = 3 * input_w_ * input_h_
    int output_size_ = 0;  // total floats in output tensor

    // Tensor names (read from engine)
    std::string input_name_;
    std::string output_name_;

    // Profiling
    float last_infer_ms_ = 0.0f;
};
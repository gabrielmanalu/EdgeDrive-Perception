#pragma once

#include <string>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <iomanip>

// ── Profiler ──────────────────────────────────────────────────────────────────
// Tracks FPS, latency, and power metrics for the inference pipeline.
// Lightweight — no external dependencies beyond STL.

class Profiler {
public:
    explicit Profiler(int window_size = 30);

    // Call at start of each frame
    void frameStart();

    // Call at end of each frame
    void frameEnd();

    // Add timing splits for this frame (ms)
    void addInferMs(float ms);
    void addPreprocessMs(float ms);
    void addTRTMs(float ms);

    // Print stats to stdout every N frames
    void printStats(int every_n_frames = 30);

    // Accessors
    float meanFPS()        const;
    float meanInferMs()    const;
    float meanPreprocessMs() const;
    float meanTRTMs()      const;
    float p99InferMs()     const;
    int   totalFrames()    const { return total_frames_; }

private:
    int window_size_;
    int total_frames_ = 0;

    std::vector<float> frame_times_ms_;    // wall time per frame
    std::vector<float> infer_times_ms_;    // total inference time
    std::vector<float> preprocess_times_ms_; // preprocessing only
    std::vector<float> trt_times_ms_;      // TRT inference only

    std::chrono::high_resolution_clock::time_point frame_start_;
};
/**
 * profiler.cpp — FPS and Latency Tracker
 * ========================================
 * Lightweight profiler for the inference pipeline.
 * Tracks wall-clock FPS, inference latency, and prints
 * periodic stats to stdout during benchmarking.
 *
 * Metrics tracked:
 *   - Wall FPS: frames per second including all overhead
 *     (preprocessing, inference, postprocessing, display)
 *   - Inference ms: pure TensorRT inference time only
 *     (from trt_engine.lastInferMs())
 *   - P99 latency: 99th percentile inference time
 *     (worst-case latency, important for real-time systems)
 *
 * Rolling window (default 30 frames) keeps stats current
 * without accumulating unbounded history.
 */

#include "profiler.hpp"

// ── Constructor ───────────────────────────────────────────────────────────────

Profiler::Profiler(int window_size)
    : window_size_(window_size)
{
    frame_times_ms_.reserve(window_size);
    infer_times_ms_.reserve(window_size);
}

// ── Frame timing ──────────────────────────────────────────────────────────────

void Profiler::frameStart() {
    frame_start_ = std::chrono::high_resolution_clock::now();
}

void Profiler::frameEnd() {
    auto now     = std::chrono::high_resolution_clock::now();
    float ms     = std::chrono::duration<float, std::milli>(
                       now - frame_start_).count();

    // Rolling window — remove oldest if full
    if (static_cast<int>(frame_times_ms_.size()) >= window_size_) {
        frame_times_ms_.erase(frame_times_ms_.begin());
    }
    frame_times_ms_.push_back(ms);

    total_frames_++;
}

void Profiler::addInferMs(float ms) {
    if (static_cast<int>(infer_times_ms_.size()) >= window_size_) {
        infer_times_ms_.erase(infer_times_ms_.begin());
    }
    infer_times_ms_.push_back(ms);
}

void Profiler::addPreprocessMs(float ms) {
    if (static_cast<int>(preprocess_times_ms_.size()) >= window_size_) {
        preprocess_times_ms_.erase(preprocess_times_ms_.begin());
    }
    preprocess_times_ms_.push_back(ms);
}

void Profiler::addTRTMs(float ms) {
    if (static_cast<int>(trt_times_ms_.size()) >= window_size_) {
        trt_times_ms_.erase(trt_times_ms_.begin());
    }
    trt_times_ms_.push_back(ms);
}

// ── Stats ─────────────────────────────────────────────────────────────────────

float Profiler::meanFPS() const {
    if (frame_times_ms_.empty()) return 0.0f;
    float mean_ms = std::accumulate(
        frame_times_ms_.begin(), frame_times_ms_.end(), 0.0f)
        / frame_times_ms_.size();
    return mean_ms > 0.0f ? 1000.0f / mean_ms : 0.0f;
}

float Profiler::meanInferMs() const {
    if (infer_times_ms_.empty()) return 0.0f;
    return std::accumulate(
        infer_times_ms_.begin(), infer_times_ms_.end(), 0.0f)
        / infer_times_ms_.size();
}

float Profiler::meanPreprocessMs() const {
    if (preprocess_times_ms_.empty()) return 0.0f;
    return std::accumulate(
        preprocess_times_ms_.begin(), preprocess_times_ms_.end(), 0.0f)
        / preprocess_times_ms_.size();
}

float Profiler::meanTRTMs() const {
    if (trt_times_ms_.empty()) return 0.0f;
    return std::accumulate(
        trt_times_ms_.begin(), trt_times_ms_.end(), 0.0f)
        / trt_times_ms_.size();
}

float Profiler::p99InferMs() const {
    if (infer_times_ms_.empty()) return 0.0f;
    std::vector<float> sorted = infer_times_ms_;
    std::sort(sorted.begin(), sorted.end());
    int idx = static_cast<int>(sorted.size() * 0.99f);
    idx = std::min(idx, static_cast<int>(sorted.size()) - 1);
    return sorted[idx];
}

// ── Print stats ───────────────────────────────────────────────────────────────

void Profiler::printStats(int every_n_frames) {
    if (total_frames_ % every_n_frames != 0) return;

    std::cout << std::fixed << std::setprecision(1);
    std::cout << "[" << std::setw(6) << total_frames_ << " frames]"
              << "  FPS: "        << std::setw(6) << meanFPS()
              << "  Total: "      << std::setw(6) << meanInferMs()    << "ms"
              << "  Pre: "        << std::setw(5) << meanPreprocessMs() << "ms"
              << "  TRT: "        << std::setw(5) << meanTRTMs()      << "ms"
              << "  P99: "        << std::setw(6) << p99InferMs()     << "ms"
              << std::endl;
}
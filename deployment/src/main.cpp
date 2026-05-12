/**
 * main.cpp — EdgeDrive Perception Entry Point
 * =============================================
 * Wires TRTEngine + YOLO26Decoder + Profiler into
 * two modes:
 *
 * Mode 1 — Benchmark (--benchmark):
 *   Loads 404 nuScenes images into RAM (no disk I/O during timing)
 *   Runs inference for N seconds
 *   Reports FPS, latency, P99
 *   Use for apples-to-apples comparison with Python baseline
 *
 * Mode 2 — Live camera (--camera):
 *   Opens USB webcam via cv::VideoCapture
 *   Runs real-time inference + display
 *   Shows FPS overlay on frame
 *   Press 'q' to quit
 *
 * Usage:
 *   # Benchmark on nuScenes images
 *   ./edgedrive --engine weights/yolo26n_det_int8.engine \
 *               --benchmark \
 *               --images /home/gabriel/EdgeDrive-Perception/test_images \
 *               --duration 60
 *
 *   # Live camera
 *   ./edgedrive --engine weights/yolo26n_det_int8.engine \
 *               --camera 0
 */

#include "trt_engine.hpp"
#include "yolo26_decoder.hpp"
#include "profiler.hpp"

#include <iostream>
#include <string>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <chrono>

#include <opencv2/opencv.hpp>

namespace fs = std::filesystem;

// ── Argument parsing ──────────────────────────────────────────────────────────

struct Args {
    std::string engine_path = "weights/yolo26n_det_int8.engine";
    std::string images_dir  = "";
    int         camera_id   = -1;
    int         duration_s  = 60;
    float       score_thresh = 0.3f;
    bool        benchmark   = false;
    bool        no_display  = false;
};

Args parseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if      (arg == "--engine"    && i+1 < argc) args.engine_path  = argv[++i];
        else if (arg == "--images"    && i+1 < argc) args.images_dir   = argv[++i];
        else if (arg == "--camera"    && i+1 < argc) args.camera_id    = std::stoi(argv[++i]);
        else if (arg == "--duration"  && i+1 < argc) args.duration_s   = std::stoi(argv[++i]);
        else if (arg == "--threshold" && i+1 < argc) args.score_thresh = std::stof(argv[++i]);
        else if (arg == "--benchmark")               args.benchmark    = true;
        else if (arg == "--no-display")              args.no_display   = true;
    }
    return args;
}

// ── Benchmark mode ────────────────────────────────────────────────────────────

void runBenchmark(TRTEngine& engine, YOLO26Decoder& decoder,
                  const std::string& images_dir, int duration_s)
{
    std::cout << "\n=== Benchmark Mode ===" << std::endl;
    std::cout << "Loading images from: " << images_dir << std::endl;

    // Step 1: Collect image paths
    std::vector<fs::path> paths;
    for (const auto& entry : fs::directory_iterator(images_dir)) {
        std::string ext = entry.path().extension().string();
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") {
            paths.push_back(entry.path());
        }
    }
    std::sort(paths.begin(), paths.end());

    if (paths.empty()) {
        std::cerr << "No images found in: " << images_dir << std::endl;
        return;
    }

    // Step 2: Preload all images into RAM
    // This eliminates disk I/O from benchmark timing — same approach
    // as production where camera frames arrive from buffer, not disk
    std::cout << "Preloading " << paths.size() << " images into RAM..." << std::endl;
    std::vector<cv::Mat> images;
    images.reserve(paths.size());
    for (const auto& p : paths) {
        cv::Mat img = cv::imread(p.string());
        if (!img.empty()) images.push_back(img);
    }
    std::cout << "Loaded " << images.size() << " images. Starting benchmark..." << std::endl;

    // Step 3: Warmup — 10 frames (TensorRT needs warm GPU cache)
    std::cout << "Warming up (10 frames)..." << std::endl;
    for (int i = 0; i < 10; i++) {
        engine.infer(images[i % images.size()]);
    }

    // Step 4: Timed benchmark loop
    Profiler profiler(100);
    auto t_start = std::chrono::high_resolution_clock::now();
    int frame_count = 0;

    std::cout << "Running for " << duration_s << " seconds..." << std::endl;

    while (true) {
        auto now = std::chrono::high_resolution_clock::now();
        float elapsed = std::chrono::duration<float>(now - t_start).count();
        if (elapsed >= duration_s) break;

        profiler.frameStart();

        const cv::Mat& frame = images[frame_count % images.size()];
        float* output = engine.infer(frame);

        auto dets = decoder.decode(
            output, frame.cols, frame.rows);

        profiler.frameEnd();
        profiler.addInferMs(engine.lastInferMs());
        profiler.addPreprocessMs(engine.lastPreprocessMs());
        profiler.addTRTMs(engine.lastTRTMs());
        profiler.printStats(100);

        frame_count++;
    }

    float total_s = std::chrono::duration<float>(
        std::chrono::high_resolution_clock::now() - t_start).count();

    // Step 5: Final results
    std::cout << "\n=== Benchmark Results ===" << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Frames        : " << frame_count << std::endl;
    std::cout << "Duration      : " << total_s << "s" << std::endl;
    std::cout << "FPS (wall)    : " << frame_count / total_s << std::endl;
    std::cout << "Infer total   : " << profiler.meanInferMs()      << "ms  (preprocess + TRT)" << std::endl;
    std::cout << "  Preprocess  : " << profiler.meanPreprocessMs() << "ms" << std::endl;
    std::cout << "  TRT only    : " << profiler.meanTRTMs()        << "ms" << std::endl;
    std::cout << "Infer P99     : " << profiler.p99InferMs()       << "ms" << std::endl;
    std::cout << "=========================" << std::endl;
}

// ── Live camera mode ──────────────────────────────────────────────────────────

void runCamera(TRTEngine& engine, YOLO26Decoder& decoder,
               int camera_id, bool no_display)
{
    std::cout << "\n=== Live Camera Mode ===" << std::endl;
    std::cout << "Opening camera " << camera_id << "..." << std::endl;

    cv::VideoCapture cap(camera_id);
    if (!cap.isOpened()) {
        throw std::runtime_error(
            "Failed to open camera: " + std::to_string(camera_id));
    }

    // Set camera resolution
    cap.set(cv::CAP_PROP_FRAME_WIDTH,  1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    std::cout << "Camera opened. Press 'q' to quit." << std::endl;

    Profiler profiler(30);
    cv::Mat  frame;

    while (true) {
        if (!cap.read(frame) || frame.empty()) {
            std::cerr << "Failed to read frame" << std::endl;
            break;
        }

        profiler.frameStart();

        // Inference
        float* output = engine.infer(frame);
        // auto   dets   = decoder.decode(output, frame.cols, frame.rows);

        auto dets = decoder.decode(output, frame.cols, frame.rows);

        // --- ADD THESE 4 LINES FOR A QUICK VISUAL SANITY CHECK ---
        if (frame_count == 0) { // Just save the very first frame
            cv::Mat annotated = decoder.draw(frame, dets);
            cv::imwrite("sanity_check_cpp.jpg", annotated);
            std::cout << ">>> Saved sanity_check_cpp.jpg to disk! <<<" << std::endl;
        }
        // ---------------------------------------------------------

        profiler.frameEnd();
        profiler.addInferMs(engine.lastInferMs());

        if (!no_display) {
            // Draw detections
            cv::Mat annotated = decoder.draw(frame, dets);

            // FPS overlay
            std::string fps_str = "FPS: " +
                std::to_string(static_cast<int>(profiler.meanFPS()));
            cv::putText(annotated, fps_str,
                        cv::Point(10, 30),
                        cv::FONT_HERSHEY_SIMPLEX, 1.0,
                        cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

            // Inference time overlay
            std::string infer_str = "Infer: " +
                std::to_string(static_cast<int>(engine.lastInferMs())) + "ms";
            cv::putText(annotated, infer_str,
                        cv::Point(10, 65),
                        cv::FONT_HERSHEY_SIMPLEX, 0.7,
                        cv::Scalar(0, 255, 0), 2, cv::LINE_AA);

            cv::imshow("EdgeDrive Perception", annotated);

            if (cv::waitKey(1) == 'q') break;
        }

        // Print stats every 30 frames
        profiler.printStats(30);
    }

    cap.release();
    if (!no_display) cv::destroyAllWindows();
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    Args args = parseArgs(argc, argv);

    std::cout << "EdgeDrive Perception — C++ TensorRT Pipeline" << std::endl;
    std::cout << "Engine : " << args.engine_path  << std::endl;
    std::cout << "Thresh : " << args.score_thresh << std::endl;

    try {
        // Initialize engine and decoder
        TRTEngine    engine(args.engine_path);
        YOLO26Decoder decoder(args.score_thresh);

        if (args.benchmark && !args.images_dir.empty()) {
            runBenchmark(engine, decoder,
                         args.images_dir, args.duration_s);
        } else if (args.camera_id >= 0) {
            runCamera(engine, decoder,
                      args.camera_id, args.no_display);
        } else {
            std::cout << "\nUsage:" << std::endl;
            std::cout << "  Benchmark: ./edgedrive "
                      << "--engine weights/yolo26n_det_int8.engine "
                      << "--benchmark "
                      << "--images /path/to/images "
                      << "--duration 60"
                      << std::endl;
            std::cout << "  Camera:    ./edgedrive "
                      << "--engine weights/yolo26n_det_int8.engine "
                      << "--camera 0"
                      << std::endl;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
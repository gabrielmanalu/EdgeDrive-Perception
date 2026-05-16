/**
 * main.cpp — EdgeDrive Perception Entry Point
 * =============================================
 * Wires TRTEngine + YOLO26Decoder + Profiler into
 * two modes:
 *
 * Mode 1 — Benchmark (--benchmark):
 *   Loads 404 nuScenes images into RAM (no disk I/O during timing)
 *   Runs inference for N seconds
 *   Reports FPS, latency split (pre/TRT), P99
 *
 * Mode 2 — Live camera (--camera / --csi):
 *   USB:  --camera 0          (cv::VideoCapture + V4L2)
 *   CSI:  --csi 0             (nvarguscamerasrc GStreamer)
 *   Runs real-time inference + HUD overlay
 *   Optional video save: --save-video output.mp4
 *   Press 'q' or ESC to quit
 *
 * Usage:
 *   # Benchmark
 *   ./edgedrive --engine weights/yolo26n_det_int8_raw.engine \
 *               --benchmark --images test_images --duration 60
 *
 *   # USB camera
 *   ./edgedrive --engine weights/yolo26n_det_int8_raw.engine \
 *               --camera 0
 *
 *   # CSI camera (Jetson)
 *   ./edgedrive --engine weights/yolo26n_det_int8_raw.engine \
 *               --csi 0
 *
 *   # Save to video
 *   ./edgedrive --engine weights/yolo26n_det_int8_raw.engine \
 *               --camera 0 --save-video demo.mp4
 *
 *   # Headless
 *   ./edgedrive --engine weights/yolo26n_det_int8_raw.engine \
 *               --camera 0 --no-display
 */

#include "trt_engine.hpp"
#include "yolo26_decoder.hpp"
#include "camera_capture.hpp"
#include "bev_visualizer.hpp"
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
    std::string engine_path  = "weights/yolo26n_det_int8_raw.engine";
    std::string images_dir   = "";
    std::string save_video   = "";
    std::string video_path   = "";
    int         camera_id    = -1;
    int         csi_id       = -1;
    int         duration_s   = 60;
    float       score_thresh = 0.3f;
    bool        benchmark    = false;
    bool        no_display   = false;
    bool        no_loop      = false;
    bool        bev          = false;
};

Args parseArgs(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if      (arg == "--engine"     && i+1 < argc) args.engine_path  = argv[++i];
        else if (arg == "--images"     && i+1 < argc) args.images_dir   = argv[++i];
        else if (arg == "--camera"     && i+1 < argc) args.camera_id    = std::stoi(argv[++i]);
        else if (arg == "--csi"        && i+1 < argc) args.csi_id       = std::stoi(argv[++i]);
        else if (arg == "--video"      && i+1 < argc) args.video_path   = argv[++i];
        else if (arg == "--save-video" && i+1 < argc) args.save_video   = argv[++i];
        else if (arg == "--duration"   && i+1 < argc) args.duration_s   = std::stoi(argv[++i]);
        else if (arg == "--threshold"  && i+1 < argc) args.score_thresh = std::stof(argv[++i]);
        else if (arg == "--benchmark")                args.benchmark    = true;
        else if (arg == "--no-display")               args.no_display   = true;
        else if (arg == "--no-loop")                  args.no_loop      = true;
        else if (arg == "--bev")                      args.bev          = true;
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

    bool debug_done = false;
    while (true) {
        auto now = std::chrono::high_resolution_clock::now();
        float elapsed = std::chrono::duration<float>(now - t_start).count();
        if (elapsed >= duration_s) break;

        profiler.frameStart();

        const cv::Mat& frame = images[frame_count % images.size()];
        float* output = engine.infer(frame);

        // Print debug info on first frame only
        // if (!debug_done) {
        //     decoder.debugOutput(output, engine.outputSize());
        //     debug_done = true;
        // }

        auto dets = decoder.decode(
            output, engine.outputSize(), frame.cols, frame.rows);
        
        // Sanity Check
        // if (frame_count == 0) { // Just save the very first frame
        //     cv::Mat annotated = decoder.draw(frame, dets);
        //     cv::imwrite("sanity_check_cpp.jpg", annotated);
        //     std::cout << ">>> Saved sanity_check_cpp.jpg to disk! <<<" << std::endl;
        // }

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
    std::cout << "  TRT only    : " << profiler.meanTRTMs()        << "ms  ← compare to Python" << std::endl;
    std::cout << "Infer P99     : " << profiler.p99InferMs()       << "ms" << std::endl;
    std::cout << "=========================" << std::endl;
}

// ── Main ──────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {
    Args args = parseArgs(argc, argv);

    std::cout << "EdgeDrive Perception — C++ TensorRT Pipeline" << std::endl;
    std::cout << "Engine : " << args.engine_path  << std::endl;
    std::cout << "Thresh : " << args.score_thresh << std::endl;

    try {
        TRTEngine     engine(args.engine_path);
        YOLO26Decoder decoder(args.score_thresh);

        if (args.benchmark && !args.images_dir.empty()) {
            runBenchmark(engine, decoder,
                         args.images_dir, args.duration_s);

        } else if (!args.video_path.empty() ||
                   args.camera_id >= 0 ||
                   args.csi_id    >= 0) {
            CameraConfig cfg;
            cfg.no_display  = args.no_display;
            cfg.save_video  = args.save_video;
            cfg.loop_video  = !args.no_loop;
            cfg.show_bev    = args.bev;

            if (!args.video_path.empty()) {
                cfg.video_path = args.video_path;
            } else if (args.csi_id >= 0) {
                cfg.gst_pipeline = buildCSIPipeline(
                    args.csi_id, cfg.width, cfg.height, cfg.fps);
            } else {
                cfg.camera_id = args.camera_id;
            }

            runCamera(engine, decoder, cfg);

        } else {
            std::cout << "\nUsage:" << std::endl;
            std::cout << "  Benchmark  : ./edgedrive --engine <path>"
                         " --benchmark --images <dir> [--duration 60]" << std::endl;
            std::cout << "  USB camera : ./edgedrive --engine <path>"
                         " --camera 0" << std::endl;
            std::cout << "  CSI camera : ./edgedrive --engine <path>"
                         " --csi 0" << std::endl;
            std::cout << "  Video file : ./edgedrive --engine <path>"
                         " --video driving.mp4 [--no-loop]" << std::endl;
            std::cout << "  Save video : ./edgedrive --engine <path>"
                         " --camera 0 --save-video demo.mp4" << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
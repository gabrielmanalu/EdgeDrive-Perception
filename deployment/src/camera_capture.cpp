/**
 * camera_capture.cpp — Live camera inference for EdgeDrive Perception
 * ====================================================================
 *
 * Supports two camera backends:
 *
 * USB camera (default):
 *   ./edgedrive --engine weights/yolo26n_det_int8_raw.engine --camera 0
 *
 * Jetson CSI camera (nvarguscamerasrc):
 *   ./edgedrive --engine weights/yolo26n_det_int8_raw.engine --csi 0
 *
 * Save output video:
 *   ./edgedrive --engine ... --camera 0 --save-video output.mp4
 *
 * Headless (no display, e.g. SSH without X11):
 *   ./edgedrive --engine ... --camera 0 --no-display
 *
 * HUD overlay (top-left):
 *   FPS   : wall-clock frames per second
 *   Pre   : preprocessing time (ms)
 *   TRT   : TensorRT inference time (ms)
 *   Dets  : detections this frame
 */

#include "camera_capture.hpp"
#include "profiler.hpp"

#include <opencv2/opencv.hpp>
#include <csignal>
#include <iomanip>
#include <iostream>
#include <sstream>

// ── Signal handler ─────────────────────────────────────────────────────────────

static volatile bool g_running = true;

static void sigintHandler(int) {
    g_running = false;
}

// ── GStreamer pipeline ─────────────────────────────────────────────────────────

std::string buildCSIPipeline(int sensor_id, int width, int height, int fps) {
    /**
     * nvarguscamerasrc → NVMM memory → nvvidconv → CPU-accessible BGR
     * This is the recommended zero-copy path on Jetson for CSI cameras.
     *
     * Equivalent shell command:
     *   gst-launch-1.0 nvarguscamerasrc sensor-id=0 \
     *     ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 \
     *     ! nvvidconv flip-method=0 \
     *     ! video/x-raw,format=BGRx \
     *     ! videoconvert \
     *     ! video/x-raw,format=BGR \
     *     ! appsink
     */
    std::ostringstream ss;
    ss << "nvarguscamerasrc sensor-id=" << sensor_id
       << " ! video/x-raw(memory:NVMM)"
       << ",width=" << width
       << ",height=" << height
       << ",framerate=" << fps << "/1"
       << " ! nvvidconv flip-method=0"
       << " ! video/x-raw,format=BGRx"
       << " ! videoconvert"
       << " ! video/x-raw,format=BGR"
       << " ! appsink drop=true max-buffers=1";
    return ss.str();
}

// ── HUD helpers ────────────────────────────────────────────────────────────────

static void drawHUD(cv::Mat& frame, const Profiler& profiler,
                    float pre_ms, float trt_ms, int det_count)
{
    // Semi-transparent dark background for HUD
    const int pad   = 8;
    const int lh    = 24;   // line height
    const int lines = 4;
    const int w     = 180;
    const int h     = lines * lh + pad * 2;

    cv::Mat overlay = frame.clone();
    cv::rectangle(overlay, cv::Rect(8, 8, w, h),
                  cv::Scalar(0, 0, 0), cv::FILLED);
    cv::addWeighted(overlay, 0.55, frame, 0.45, 0, frame);

    auto hud_line = [&](int line, const std::string& key,
                        const std::string& val, cv::Scalar color) {
        int y = 8 + pad + lh * line + lh - 6;
        cv::putText(frame, key, cv::Point(14, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.52,
                    cv::Scalar(180, 180, 180), 1, cv::LINE_AA);
        cv::putText(frame, val, cv::Point(74, y),
                    cv::FONT_HERSHEY_SIMPLEX, 0.52,
                    color, 1, cv::LINE_AA);
    };

    auto fmt1 = [](float v, const std::string& unit) {
        std::ostringstream ss;
        ss << std::fixed << std::setprecision(1) << v << unit;
        return ss.str();
    };

    float fps = profiler.meanFPS();
    cv::Scalar fps_color = fps >= 25.0f
        ? cv::Scalar(80, 220, 80)    // green
        : fps >= 15.0f
        ? cv::Scalar(80, 200, 220)   // yellow
        : cv::Scalar(80, 80, 220);   // red

    hud_line(0, "FPS  :", fmt1(fps, ""),       fps_color);
    hud_line(1, "Pre  :", fmt1(pre_ms, "ms"),  cv::Scalar(200, 200, 200));
    hud_line(2, "TRT  :", fmt1(trt_ms, "ms"),  cv::Scalar(200, 200, 200));
    hud_line(3, "Dets :", std::to_string(det_count), cv::Scalar(200, 200, 200));
}

// ── runCamera ──────────────────────────────────────────────────────────────────

void runCamera(TRTEngine& engine, YOLO26Decoder& decoder,
               const CameraConfig& cfg)
{
    std::signal(SIGINT, sigintHandler);

    // ── Open capture ──────────────────────────────────────────────────────────

    cv::VideoCapture cap;

    if (!cfg.video_path.empty()) {
        std::cout << "Opening video file: " << cfg.video_path << std::endl;
        cap.open(cfg.video_path);
        if (!cap.isOpened())
            throw std::runtime_error("Failed to open video: " + cfg.video_path);

    } else if (!cfg.gst_pipeline.empty()) {
        std::cout << "Opening CSI camera via GStreamer..." << std::endl;
        std::cout << "Pipeline: " << cfg.gst_pipeline << std::endl;
        cap.open(cfg.gst_pipeline, cv::CAP_GSTREAMER);
    } else {
        std::cout << "Opening USB camera " << cfg.camera_id << "..." << std::endl;
        cap.open(cfg.camera_id, cv::CAP_V4L2);
        if (cap.isOpened()) {
            cap.set(cv::CAP_PROP_FOURCC,
                        cv::VideoWriter::fourcc('M','J','P','G'));
            cap.set(cv::CAP_PROP_FRAME_WIDTH,  cfg.width);
            cap.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);
            cap.set(cv::CAP_PROP_FPS,          cfg.fps);
            // Minimize buffer lag — keep only the latest frame
            cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
        }
    }

    if (!cap.isOpened()) {
        throw std::runtime_error(
            "Failed to open camera. Check connection and try:\n"
            "  USB: --camera 0  (or 1, 2 for other devices)\n"
            "  CSI: --csi 0     (uses nvarguscamerasrc)");
    }

    // Report actual capture properties
    int actual_w   = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int actual_h   = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double actual_fps = cap.get(cv::CAP_PROP_FPS);
    std::cout << "Camera opened: "
              << actual_w << "x" << actual_h
              << " @ " << actual_fps << " FPS" << std::endl;

    // ── Optional video writer ─────────────────────────────────────────────────

    cv::VideoWriter writer;
    if (!cfg.save_video.empty()) {
        int fourcc = cv::VideoWriter::fourcc('m','p','4','v');
        writer.open(cfg.save_video, fourcc,
                    actual_fps > 0 ? actual_fps : 30.0,
                    cv::Size(actual_w, actual_h));
        if (!writer.isOpened()) {
            std::cerr << "Warning: could not open video writer at "
                      << cfg.save_video << std::endl;
        } else {
            std::cout << "Saving video to: " << cfg.save_video << std::endl;
        }
    }

    // ── Warmup ────────────────────────────────────────────────────────────────

    std::cout << "Warming up (" << cfg.warmup_frames << " frames)..." << std::endl;
    cv::Mat warmup_frame;
    for (int i = 0; i < cfg.warmup_frames && g_running; i++) {
        cap.read(warmup_frame);
        if (!warmup_frame.empty()) {
            engine.infer(warmup_frame);
        }
    }

    // ── Inference loop ────────────────────────────────────────────────────────

    std::cout << "\n=== Live Camera Mode ===" << std::endl;
    if (!cfg.no_display)
        std::cout << "Press 'q' to quit." << std::endl;
    else
        std::cout << "Headless mode. Press Ctrl+C to stop." << std::endl;

    Profiler profiler(30);
    cv::Mat  frame;
    int      frame_count = 0;

    while (g_running) {

        // Grab latest frame
        if (!cap.read(frame) || frame.empty()) {
            // End of video file — loop if enabled
            if (!cfg.video_path.empty() && cfg.loop_video) {
                cap.set(cv::CAP_PROP_POS_FRAMES, 0);
                cap.read(frame);
                if (frame.empty()) break;
            } else {
                std::cerr << (cfg.video_path.empty()
                    ? "Warning: empty frame — camera disconnected?"
                    : "Video ended.") << std::endl;
                break;
            }
        }

        profiler.frameStart();

        // ── Inference ─────────────────────────────────────────────────────────
        float* output = engine.infer(frame);
        auto   dets   = decoder.decode(
            output, engine.outputSize(), frame.cols, frame.rows);

        profiler.frameEnd();
        profiler.addInferMs(engine.lastInferMs());
        profiler.addPreprocessMs(engine.lastPreprocessMs());
        profiler.addTRTMs(engine.lastTRTMs());

        frame_count++;

        // ── Display / save ────────────────────────────────────────────────────
        if (!cfg.no_display || writer.isOpened()) {
            cv::Mat annotated = decoder.draw(frame, dets);
            drawHUD(annotated, profiler,
                    engine.lastPreprocessMs(),
                    engine.lastTRTMs(),
                    static_cast<int>(dets.size()));

            if (!cfg.no_display) {
                cv::imshow("EdgeDrive Perception", annotated);
                int key = cv::waitKey(1);
                if (key == 'q' || key == 27) break;  // q or ESC
            }

            if (writer.isOpened())
                writer.write(annotated);
        }

        // Console stats every 30 frames
        profiler.printStats(30);
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────

    cap.release();
    if (writer.isOpened()) writer.release();
    if (!cfg.no_display)   cv::destroyAllWindows();

    std::cout << "\n=== Camera Session Summary ===" << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    std::cout << "Frames processed : " << frame_count       << std::endl;
    std::cout << "Mean FPS         : " << profiler.meanFPS() << std::endl;
    std::cout << "Mean preprocess  : " << profiler.meanPreprocessMs() << "ms" << std::endl;
    std::cout << "Mean TRT         : " << profiler.meanTRTMs()        << "ms" << std::endl;
    std::cout << "==============================" << std::endl;
}
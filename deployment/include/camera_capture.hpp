#pragma once

#include "trt_engine.hpp"
#include "yolo26_decoder.hpp"
#include <string>

/**
 * CameraConfig — runtime options for live camera mode
 */
struct CameraConfig {
    int         camera_id    = 0;        // USB camera index (default: /dev/video0)
    std::string gst_pipeline = "";       // GStreamer pipeline (overrides camera_id)
    std::string video_path   = "";       // pre-recorded video file (overrides camera)
    int         width        = 1280;     // capture width
    int         height       = 720;      // capture height
    int         fps          = 30;       // target capture FPS
    bool        loop_video   = true;     // loop video file when it ends
    bool        no_display   = false;    // headless mode (no imshow)
    std::string save_video   = "";       // path to save output video (empty = no save)
    int         warmup_frames = 10;      // discard first N frames
    bool        show_bev     = false;    // show bird's eye view
};

/**
 * Build a GStreamer pipeline string for Jetson CSI camera.
 * Uses nvarguscamerasrc + nvvidconv for zero-copy NVMM path.
 *
 * Example:
 *   std::string pipeline = buildCSIPipeline(0, 1280, 720, 30);
 *   CameraConfig cfg;
 *   cfg.gst_pipeline = pipeline;
 */
std::string buildCSIPipeline(int sensor_id, int width, int height, int fps);

/**
 * Build a GStreamer pipeline for hardware-accelerated video file decode.
 * Uses nvv4l2decoder (NVDEC) — eliminates ~3-4ms software decode per frame.
 * Auto-detects H.265 from filename, defaults to H.264.
 * Used automatically when --video is specified; falls back to software decode.
 */
std::string buildVideoNVDECPipeline(const std::string& path);

/**
 * Run live camera inference loop.
 * Blocks until user presses 'q' or SIGINT.
 *
 * @param engine    Initialized TRTEngine
 * @param decoder   Initialized YOLO26Decoder
 * @param cfg       Camera configuration
 */
void runCamera(TRTEngine& engine, YOLO26Decoder& decoder,
               const CameraConfig& cfg);
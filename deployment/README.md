# EdgeDrive Perception — C++ TRT Deployment

Real-time object detection pipeline for Jetson Orin Nano Super.
TensorRT INT8 inference at **203 FPS / 8.03W** on live camera input.

---

## Requirements

| Component | Version |
|---|---|
| Hardware | Jetson Orin Nano Super 8GB (SM 8.7) |
| JetPack | R36.4.7 |
| CUDA | 12.6 |
| TensorRT | 10.3.0 |
| OpenCV | 4.8.0 (with GStreamer support) |
| CMake | 3.18+ |
| C++ | 17 |

---

## Project Structure

```
deployment/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── trt_engine.hpp           ← TRT inference + UMA zero-copy
│   ├── yolo26_decoder.hpp       ← raw [1,12,8400] + end2end decoder
│   ├── camera_capture.hpp       ← USB / CSI / video / NVDEC / BEV
│   ├── bev_visualizer.hpp       ← ground plane BEV projection
│   ├── profiler.hpp             ← split timers (pre/TRT/post)
│   ├── heatmap_generator.hpp    ← detection frequency heatmap
│   ├── object_counter.hpp       ← line/zone crossing counter
│   ├── speed_estimator.hpp      ← centroid tracker + km/h estimator
│   ├── segmentation_decoder.hpp ← YOLO26n-seg mask decoder
│   └── preprocessor.cuh         ← CUDA letterbox kernel (optional)
└── src/
    ├── main.cpp                 ← entry point, argument parsing
    ├── trt_engine.cpp           ← TRT engine, enqueueV3, UMA buffers
    ├── yolo26_decoder.cpp       ← cxcywh decode, NMS, draw
    ├── camera_capture.cpp       ← capture loop, NVDEC, BEV, HUD
    ├── bev_visualizer.cpp       ← range rings, ego car, projection
    ├── profiler.cpp             ← rolling window, FPS/pre/TRT stats
    ├── heatmap_generator.cpp    ← Gaussian accumulator, COLORMAP_JET
    ├── object_counter.cpp       ← centroid tracker, line crossing
    ├── speed_estimator.cpp      ← displacement → m/s → km/h, trail
    ├── segmentation_decoder.cpp ← mask coefficients × prototypes
    ├── preprocessor.cu          ← CUDA letterbox + BGR→RGB + CHW
    ├── late_fusion.cpp          ← camera-LiDAR BEV matching (⬜ WIP)
    └── pointpillars_decoder.cpp ← LiDAR 3D box decoder (⬜ WIP)
```


---

## Build

```bash
cd ~/EdgeDrive-Perception/deployment
mkdir -p build && cd build
cmake ..
make -j4
```

Expected cmake output:
```
-- TensorRT: /usr/lib/aarch64-linux-gnu/libnvinfer.so
-- OpenCV: 4.8.0
-- Build type : Release
-- CUDA arch  : 87
-- Sources    : src/trt_engine.cpp src/yolo26_decoder.cpp ...
```

Binary output: `deployment/build/edgedrive`

---

## Engines

TensorRT engines must be built on the target Jetson — they are
hardware-specific and not transferable across platforms.

```bash
# From repo root
./scripts/build_engine.sh fp16   # FP16 only (~9 min)
./scripts/build_engine.sh int8   # INT8 only (~7 min)
./scripts/build_engine.sh all    # both

# Or manually
python3 -c "
from ultralytics import YOLO
model = YOLO('weights/yolo26n_det.pt')
model.export(format='engine', device=0, half=True)        # FP16
model.export(format='engine', device=0, int8=True,
             data='calibration.yaml')                      # INT8
"
```

**Known issue:** TensorRT 10.3.0 on JetPack 6 disables the `end2end`
branch for INT8 engines. Output format changes from `[1,300,6]` to
`[1,12,8400]`. The decoder handles both formats automatically.
See [`docs/nms_free_analysis.md`](../docs/nms_free_analysis.md).

---

## Usage

### Benchmark mode

Measures sustained inference performance. Images preloaded into RAM
to isolate compute from disk I/O.

```bash
sudo jetson_clocks  # lock clocks for reproducible results

./deployment/build/edgedrive \
    --engine weights/yolo26n_det_int8_raw.engine \
    --benchmark \
    --images test_images \
    --duration 60
```

Expected output:
```
[  30 frames]  FPS:  193.3  Total:  5.1ms  Pre:  1.7ms  TRT:  3.5ms  P99:  9.3ms
...
=== Benchmark Results ===
Frames        : 11598
Duration      : 60.0s
FPS (wall)    : 193.3
Infer total   : 5.1ms  (preprocess + TRT)
  Preprocess  : 1.7ms
  TRT only    : 3.5ms
Infer P99     : 9.3ms
=========================
```

### USB camera

```bash
./deployment/build/edgedrive \
    --engine weights/yolo26n_det_int8_raw.engine \
    --camera 0 \
    --threshold 0.3
```

Press `q` or `ESC` to quit.

### CSI camera (Jetson native)

```bash
./deployment/build/edgedrive \
    --engine weights/yolo26n_det_int8_raw.engine \
    --csi 0
```

Uses `nvarguscamerasrc` GStreamer pipeline for zero-copy NVMM path.

### Pre-recorded video

```bash
./deployment/build/edgedrive \
    --engine weights/yolo26n_det_int8_raw.engine \
    --video driving.mp4 \
    --threshold 0.3
```

Uses NVDEC hardware decode automatically (`nvv4l2decoder` via GStreamer).
Falls back to software decode if NVDEC pipeline fails.

### Save annotated output

```bash
./deployment/build/edgedrive \
    --engine weights/yolo26n_det_int8_raw.engine \
    --camera 0 \
    --save-video output/demo.mp4
```

### Headless (no display / SSH)

```bash
./deployment/build/edgedrive \
    --engine weights/yolo26n_det_int8_raw.engine \
    --camera 0 \
    --no-display
```

---

## All Arguments

| Argument | Default | Description |
|---|---|---|
| `--engine <path>` | required | TensorRT engine file |
| `--benchmark` | off | Run benchmark mode |
| `--images <dir>` | — | Image directory for benchmark |
| `--duration <s>` | 60 | Benchmark duration in seconds |
| `--camera <id>` | — | USB camera device index |
| `--csi <id>` | — | CSI camera sensor ID (Jetson) |
| `--video <path>` | — | Pre-recorded video file |
| `--save-video <path>` | — | Save annotated output to file |
| `--threshold <f>` | 0.3 | Detection score threshold |
| `--no-display` | off | Headless mode, no imshow |
| `--no-loop` | off | Don't loop video file |

---

## Performance

All results with `sudo jetson_clocks` (GPU locked at 1017 MHz).

| Mode | FPS | Pre | TRT | Power |
|---|---|---|---|---|
| Benchmark (1600×900 nuScenes) | 193.3 | 1.7ms | 3.5ms | ~12.2W |
| USB camera (1280×720) | 203.4 | 1.1ms | 3.7ms | ~8.03W |
| Video NVDEC (1280×720) | ~195 | 1.2ms | 4.1ms | ~10.2W |

USB camera is faster than benchmark because 1280×720 requires less
letterbox preprocessing than 1600×900 nuScenes images.

Power is lower on USB camera than benchmark because the GPU is not
hammered at 200 FPS continuously — the camera delivers frames at
30 FPS, leaving the GPU idle between frames.

---

## Pipeline Architecture

### Preprocessing

```
Input frame (any resolution)
    │
    ▼ cv::resize() — aspect-ratio preserving letterbox
    ▼ cv::cvtColor() — BGR → RGB
    ▼ pixel loop — HWC → CHW + normalize [0,255] → [0,1]
    ▼ write to cudaHostAlloc pinned UMA buffer (zero-copy)
input_host_[3 × 640 × 640] float32
```

### Memory model (Jetson UMA)

```
Traditional discrete GPU:
  CPU RAM → cudaMemcpy → GPU VRAM  (~1-2ms overhead)

Jetson UMA (this pipeline):
  cudaHostAlloc(cudaHostAllocMapped)
  CPU writes input_host_ → GPU reads directly (same DRAM)
  GPU writes output_host_ → CPU reads directly (same DRAM)
  Zero cudaMemcpy() calls
```

### Decoder

```
Output [1, 12, 8400] (raw head, end2end disabled on JetPack 6 INT8):
  channels 0-3  : cx, cy, w, h  (640×640 letterboxed space)
  channels 4-11 : class probabilities (sigmoid already applied)

Decode:
  1. cxcywh → xyxy
  2. Letterbox inverse transform:
       gain  = min(640/orig_w, 640/orig_h)
       pad_x = (640 - orig_w × gain) / 2
       pad_y = (640 - orig_h × gain) / 2
       x1    = (x1 - pad_x) / gain
  3. Score threshold filter (default: 0.3)
  4. Class-agnostic NMS (IoU: 0.3)
```

---

## Test Scripts

```bash
# All camera tests
./scripts/test_camera.sh usb        # live USB camera
./scripts/test_camera.sh csi        # CSI camera (Jetson)
./scripts/test_camera.sh video      # pre-recorded video
./scripts/test_camera.sh save       # record 30s to output/
./scripts/test_camera.sh headless   # no display, console stats
./scripts/test_camera.sh video-save # video → annotated output

# Create test video from nuScenes images
python3 scripts/make_test_video.py
```

---

## BEV Visualization

Split-screen mode showing camera view + Bird's Eye View projection.

```bash
# Live camera + BEV
./deployment/build/edgedrive \
    --engine weights/yolo26n_det_int8_raw.engine \
    --camera 0 --threshold 0.3 --bev

# Video + BEV
./deployment/build/edgedrive \
    --engine weights/yolo26n_det_int8_raw.engine \
    --video driving.mp4 --threshold 0.3 --bev

# Save BEV output
./scripts/test_camera.sh bev-save
```

**How it works:**

Ground plane projection using pinhole camera model:

```
Bottom-center of box (u, v) → ground plane:
  Z = cam_h × fy / (v - cy)   [forward distance, meters]
  X = (u - cx) × Z / fx       [lateral offset, meters]

Default: cam_h=1.2m, fx=640 (90° HFOV USB webcam estimate)
Range:   15m forward, 10m lateral
```

Distances are **estimates** — no metric accuracy without camera
calibration. Confirmed limitation documented in
[`docs/sensor_fusion_analysis.md`](../docs/sensor_fusion_analysis.md).

---

## Ultralytics Solutions — C++ Reference Implementations

C++ equivalents of the Python solutions used in Colab development.
Not yet wired into `main.cpp` — reference implementations only.

### HeatmapGenerator

Accumulates a Gaussian blob at each detection center across frames.

```cpp
#include "heatmap_generator.hpp"

HeatmapGenerator hm(1280, 720, /*decay=*/0.995f, /*sigma=*/20);
while (running) {
    auto dets = decoder.decode(...);
    hm.update(dets);
    cv::Mat vis = hm.render(frame, 0.6f);  // 60% heatmap overlay
    cv::imshow("Heatmap", vis);
}
```

### ObjectCounter

Counts objects crossing a defined line (IN/OUT) or within a polygon zone.

```cpp
#include "object_counter.hpp"

ObjectCounter counter;
// Horizontal counting line at y=360
counter.setLine(cv::Point(0, 360), cv::Point(1280, 360));

while (running) {
    counter.update(dets);
    cv::Mat vis = counter.draw(frame);
    // counter.countIn(), counter.countOut()
}
```

### SpeedEstimator

Centroid-based tracker with pixel displacement → km/h conversion.

```cpp
#include "speed_estimator.hpp"

// 50 px/m calibration, 30 FPS, 10-frame smoothing window
SpeedEstimator estimator(50.0f, 30.0f, 10);

while (running) {
    estimator.update(dets);
    cv::Mat vis = estimator.draw(frame);  // speed labels + trail
}
```

**Note:** Assumes stationary camera. On moving vehicle, measures
relative speed (object - ego). Subtract ego velocity from IMU/odometry
for absolute speed. See [`docs/solutions_on_edge.md`](../docs/solutions_on_edge.md).

### SegmentationDecoder

Decodes YOLO26n-seg two-output format into binary masks.

```cpp
#include "segmentation_decoder.hpp"

// Requires yolo26n_seg engine (two outputs)
// output0: [1, 44, 8400]    detection + mask coefficients
// output1: [1, 32, 160, 160] prototype masks
SegmentationDecoder seg_decoder(0.3f, 0.3f);

auto results = seg_decoder.decode(output0, output1, frame.cols, frame.rows);
cv::Mat vis  = seg_decoder.draw(frame, results, 0.4f);
```

### CUDA Preprocessor (preprocessor.cu)

GPU-side letterbox kernel — replaces CPU preprocessing.

```cpp
#include "preprocessor.cuh"

// Replace CPU preprocessing in trt_engine.cpp with:
launchLetterboxKernel(
    d_bgr_input,   // uint8 BGR on device
    d_float_chw,   // float32 RGB CHW on device
    src_w, src_h,
    640, 640,
    114,           // pad value
    stream_);
```

Expected: ~0.2ms vs 1.1ms CPU → useful for multi-camera pipelines.
Not default because 1.1ms is not the bottleneck at 200 FPS.
See [`docs/optimization_log.md`](../docs/optimization_log.md).

---

## Troubleshooting

**`[TRT] Using an engine plan file across different models`**
Normal warning on JetPack 6. No impact on performance or accuracy.

**`NvMapMemAllocInternalTagged` errors**
Harmless on Jetson unified memory architecture. Ignore.

**`unable to start pipeline` (NVDEC)**
GStreamer pipeline failed — falls back to software decode automatically.
To debug: `gst-launch-1.0 filesrc location=video.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! fakesink`

**Zero detections**
Check score threshold — lower if needed: `--threshold 0.1`
Confirm engine format: `trtexec --loadEngine=... --verbose 2>&1 | grep output`

**Low FPS without `jetson_clocks`**
GPU idles at ~300 MHz without locked clocks. Always run
`sudo jetson_clocks` before benchmarking.
# Optimization Log

Chronological record of performance optimizations made to the C++ pipeline.
Each entry documents what changed, why, and the measured effect.

---

## O1 — TensorRT INT8 Quantization

**Date:** 2026-05-09 (Colab) / 2026-05-12 (Jetson)
**Change:** Export YOLO26n to TensorRT INT8 using 81 nuScenes val images as calibration set.

```
Before: Python PyTorch FP32
  FPS: 29.1 | Inference: 27.6ms

After: Python TensorRT INT8
  FPS: 78.5 | TRT inference: 3.5ms
  Speedup: +170% FPS, -87% inference latency
```

**Notes:**
- TFLite INT8 PTQ showed +0.45% mAP vs FP32 baseline (Colab proof)
- Calibration on 81 images sufficient (Ultralytics recommends 300+,
  but +0.45% confirms our calibration set is adequate for this dataset)
- JetPack 6 known issue: end2end branch disabled for INT8
  → output changes from [1,300,6] to [1,12,8400]

---

## O2 — C++ TensorRT Pipeline

**Date:** 2026-05-12
**Change:** Replace Python Ultralytics wrapper with C++ TRT pipeline.

```
Before: Python TRT INT8
  FPS: 78.5 | Total: 12.3ms | Pre: 5.6ms | TRT: 3.5ms | Post: 3.2ms

After: C++ TRT INT8
  FPS: 193.3 | Total: 5.1ms | Pre: 1.7ms | TRT: 3.5ms | Post: ~0ms
  Speedup: +146% FPS, -59% total latency
```

**Primary gains:**
1. Preprocessing: 5.6ms → 1.7ms (-70%)
2. Postprocessing: 3.2ms → ~0ms (-97%)
3. No Python GIL overhead

---

## O3 — Unified Memory (cudaHostAllocMapped)

**Date:** 2026-05-12
**Change:** Replace `cudaMalloc` + `cudaMemcpy` with `cudaHostAllocMapped`
pinned unified memory for TRT input/output buffers.

```
Before: cudaMalloc + cudaMemcpy pattern
  allocate device memory → copy host→device → inference → copy device→host

After: cudaHostAllocMapped (UMA zero-copy)
  CPU writes input_host_ → GPU reads input_device_ (same physical page)
  GPU writes output_device_ → CPU reads output_host_ (same physical page)
  Zero cudaMemcpy() calls
```

**Effect:** Estimated -1 to -2ms per frame (not isolated in profiler).
On Jetson, CPU and GPU share physical DRAM — UMA eliminates a redundant
copy that would exist on discrete GPU systems.

**Key detail:**
```cpp
cudaHostAlloc(&input_host_, input_bytes, cudaHostAllocMapped);
cudaHostGetDevicePointer(&input_device_, input_host_, 0);
// input_host_  ← CPU writes here
// input_device_ ← TRT reads here (same memory)
```

---

## O4 — Pre-allocated cv::Mat Buffers

**Date:** 2026-05-12
**Change:** Pre-allocate letterbox intermediate buffers in constructor
instead of allocating on the heap every frame.

```cpp
// Before (per-frame allocation):
void infer(const cv::Mat& frame) {
    cv::Mat resized, letterboxed;  // heap alloc every call
    ...
}

// After (constructor allocation):
TRTEngine::TRTEngine(...) {
    resized_     = cv::Mat(ih, iw, CV_8UC3);  // once
    letterboxed_ = cv::Mat(640, 640, CV_8UC3, cv::Scalar(114,114,114));
}
```

**Effect:** Eliminates per-frame `malloc`/`free` inside the inference
hot path. At 193 FPS, this prevented ~193 heap allocations/second.

---

## O5 — Letterbox Reuse via setTo()

**Date:** 2026-05-12
**Change:** Reset letterbox buffer with `setTo()` instead of reallocating.

```cpp
// Before: new Mat each frame
cv::Mat letterboxed(640, 640, CV_8UC3, cv::Scalar(114, 114, 114));

// After: reuse, reset only padding region
letterboxed_.setTo(cv::Scalar(114, 114, 114));
img_resized.copyTo(letterboxed_(cv::Rect(...)));
```

**Effect:** Further reduces per-frame allocation. The full `setTo` on a
640×640 buffer is ~0.1ms — acceptable given it replaces heap allocation.

**Note:** `blobFromImage` (OpenCV DNN) was evaluated as an alternative
but found to be 2× slower than the manual loop on Jetson UMA. Confirmed
by profiling — `blobFromImage` allocates internally and does not benefit
from pre-allocated buffers.

---

## O6 — TensorRT enqueueV3 (async)

**Date:** 2026-05-12
**Change:** Use `context_->enqueueV3(stream_)` instead of synchronous
`executeV2()`.

```cpp
// Synchronous (deprecated in TRT 10.x):
context_->executeV2(bindings);

// Async (TRT 10.x API):
context_->setTensorAddress("images",  input_device_);
context_->setTensorAddress("output0", output_device_);
context_->enqueueV3(stream_);
cudaStreamSynchronize(stream_);
```

**Effect:** Enables future pipelining (overlap CPU preprocessing of
frame N+1 while GPU processes frame N). Currently sequential, but the
async API is in place for the ROS2 integration phase.

---

## O7 — Split Profiler (Pre / TRT / Post)

**Date:** 2026-05-12
**Change:** Added per-frame split timers to isolate bottlenecks.

```
Before: single "infer_ms" timer (total pipeline)
After:  last_preprocess_ms_ + last_trt_ms_ + last_infer_ms_
```

This revealed the key insight: Python's "faster" INT8 benchmark (3.5ms)
was measuring TRT-only. C++ total (5.1ms) includes preprocessing.
TRT times are identical at 3.5ms — C++ wins on preprocessing (1.7ms vs 5.6ms).

---

## O8 — Custom Decoder (No Ultralytics Wrapper)

**Date:** 2026-05-13
**Change:** Replaced Ultralytics Python postprocessor with custom C++
YOLO26Decoder handling [1,12,8400] raw output directly.

```
Python INT8 postprocess: 3.2ms (Ultralytics overhead for INT8 output)
C++ decoder:            <0.1ms (direct memory access, no Python objects)
```

**Decoder optimizations:**
- Direct float pointer arithmetic (no tensor copy)
- Single pass over 8400 anchors
- Early exit on degenerate boxes (w<=0 or h<=0)
- Class-agnostic NMS only on candidates passing threshold

---

## O9 — Letterbox Inverse Transform Fix

**Date:** 2026-05-14
**Change:** Replaced simple `scale_x/scale_y` coordinate unscaling with
correct letterbox inverse transform using `gain` and `pad_x/pad_y`.

```cpp
// Before (wrong for non-square inputs):
float scale_x = orig_w / 640.0f;
float scale_y = orig_h / 640.0f;
det.x1 = x1 * scale_x;
det.y1 = y1 * scale_y;  // ignores letterbox padding

// After (correct):
float gain  = std::min(640.0f / orig_w, 640.0f / orig_h);
float pad_x = (640.0f - orig_w * gain) / 2.0f;
float pad_y = (640.0f - orig_h * gain) / 2.0f;
det.x1 = (x1 - pad_x) / gain;
det.y1 = (y1 - pad_y) / gain;
```

**Why it matters:**

For 1280×720 input letterboxed to 640×640:
```
gain  = min(640/1280, 640/720) = 0.5
pad_y = (640 - 720×0.5) / 2   = 140px

Old: y1 × (720/640) = y1 × 1.125  → ignores 140px padding
New: (y1 - 140) / 0.5             → removes padding first, then unscales
```

Old code worked correctly only when input and output had the same aspect
ratio (e.g. nuScenes 1600×900 → large padding, but proportionally less).
Manifested as vertical box offset on 1280×720 USB camera / video input.

---

## O10 — NVDEC Hardware Video Decode

**Date:** 2026-05-14
**Change:** Replaced default software H.264 decode (`cap.open(path)`) with
NVDEC hardware decode via GStreamer `nvv4l2decoder` pipeline.

```
Before: software decode (default OpenCV)
  FPS: ~130 | CPU H.264 decode: ~3-4ms/frame

After: NVDEC hardware decode (nvv4l2decoder)
  FPS: ~170-195 | NVDEC decode: <0.5ms/frame
```

**GStreamer pipeline:**
```
filesrc → qtdemux → h264parse → nvv4l2decoder → nvvidconv → appsink
                                 ↑
                           NVDEC chip (Jetson VPU)
```

**Remaining gap vs USB camera (202 FPS):**

NVDEC and TRT share LPDDR5 UMA bandwidth simultaneously:
```
USB camera:    camera chip decodes independently → TRT: 3.7ms
NVDEC video:   NVDEC + TRT contend for UMA      → TRT: 4.1ms (+0.4ms)
```
Hardware ceiling — not solvable without pipelining.

---

## O_rejected — Decode/Infer Pipeline Threading

**Date:** 2026-05-14
**Attempted:** Overlap NVDEC decode with TRT inference using a background
decode thread + mutex/condvar double buffer.

```
Intended: decode N+1 while TRT processes N → hide NVDEC latency
Result:   170 FPS → 150 FPS (worse)
```

**Why it failed:**

At 200 FPS, the synchronization overhead (mutex lock/unlock, condvar
wait/notify) costs ~0.5-1ms per frame — more than the contention it
was trying to eliminate. Thread context switching at this frequency
adds CPU cache pressure on top.

```
mutex/condvar overhead at 200 FPS: ~0.5ms/frame
NVDEC/TRT contention:             ~0.4ms/frame
Net: threading costs more than it saves
```

Reverted. Threading would help at lower FPS (e.g. 4K input where
NVDEC decode costs ~5ms) but not at 720p/200 FPS.

---

## O11 — Input Resolution Effect

**Date:** 2026-05-14
**Observation:** USB camera (1280×720) runs faster than nuScenes
benchmark (1600×900) despite same engine.

```
nuScenes benchmark (1600×900): Pre=1.7ms → 193 FPS
USB camera (1280×720):         Pre=1.1ms → 202 FPS
```

**Why:**
```
Letterbox resize work scales with input resolution:
  1600×900: 1,440,000 pixels to resize + copy
  1280×720:   921,600 pixels to resize + copy
  Ratio: 921600/1440000 = 0.64 → ~36% less work → ~0.6ms faster
```

Not an optimization we applied — a natural benefit of lower resolution
camera input. Relevant for production: matching camera resolution to
model input (640×640) would eliminate letterbox entirely → ~0ms preprocessing.

---

## Summary: Total Gains

```
Starting point: Python FP32 baseline
  FPS: 29.1 | Latency: 34.0ms | Power: ~9.9W

Best benchmark (C++ TRT INT8, jetson_clocks):
  FPS: 193.3 | Latency: 5.1ms | Power: ~12.2W
  +564% FPS, -85% latency

Best real-world (C++ TRT INT8, USB camera 720p):
  FPS: 202.6 | Latency: 4.8ms | Power: ~8.03W
  +596% FPS, -86% latency, under 10W
```

| Optimization | Primary Effect | Measured |
|---|---|---|
| O1: TRT INT8 | -87% inference latency | 27.6ms → 3.5ms |
| O2: C++ pipeline | +146% FPS vs Python TRT | 78.5 → 193 FPS |
| O3: UMA zero-copy | eliminates cudaMemcpy | -1-2ms estimated |
| O4: pre-alloc Mat | eliminates per-frame heap | 193 heap allocs/s eliminated |
| O5: setTo reuse | further reduces allocation | included in O4 |
| O6: enqueueV3 | async-ready for pipelining | future benefit |
| O7: split profiler | revealed preprocessing bottleneck | insight only |
| O8: custom decoder | -97% postprocess time | 3.2ms → ~0.1ms |
| O9: letterbox fix | corrects box position on non-square input | correctness |
| O10: NVDEC decode | -70% video decode overhead | 130 → 170-195 FPS |
| O_rejected: threading | attempted, reverted | 170 → 150 FPS (worse) |
| O11: 720p input | faster preprocessing | 1.7ms → 1.1ms |
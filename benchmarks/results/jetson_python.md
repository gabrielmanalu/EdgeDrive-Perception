# Jetson Python Benchmarks

**Hardware:** Jetson Orin Nano Super 8GB | JetPack R36.4.7 | CUDA 12.6 | TensorRT 10.3.0
**Clocks:** `sudo jetson_clocks` (CPU 1728 MHz, GPU 1017 MHz)
**Dataset:** 404 nuScenes Mini CAM_FRONT images, preloaded into RAM
**Duration:** 60 seconds sustained run

---

## 404 Images — 60s Sustained Run

### FP32 (PyTorch)

```
FPS (wall)    : 29.1
Preprocess    : 5.2ms
Inference     : 27.6ms
Postprocess   : 1.1ms
Total / frame : 34.0ms
```

### TensorRT FP16

```
FPS (wall)    : 87.6
Preprocess    : 5.6ms
Inference     : 4.2ms  ← TRT GPU time
Postprocess   : 1.2ms
Total / frame : 11.0ms
Engine size   : 8.0 MB
```

### TensorRT INT8

```
FPS (wall)    : 78.5
Preprocess    : 5.6ms
Inference     : 3.5ms  ← TRT GPU time (faster than FP16)
Postprocess   : 3.2ms  ← 2.7× slower than FP16 postprocess
Total / frame : 12.3ms
Engine size   : 5.4 MB
```

**Note:** Python INT8 is slower than FP16 despite faster TRT inference.
Ultralytics postprocessor handles INT8 output tensor differently,
adding ~2ms overhead. This is a Python runtime issue, not a model issue.
The C++ decoder handles INT8 output directly with no such penalty.

---

## Single Image — 20 Iterations

### FP32 (PyTorch)

```
FPS (wall)    : 22.6
Preprocess    : 6.3ms
Inference     : 28.7ms
Postprocess   : 0.9ms
Total / frame : 35.9ms
```

### TensorRT FP16

```
FPS (wall)    : 50.3
Preprocess    : 6.7ms
Inference     : 4.5ms
Postprocess   : 1.3ms
Total / frame : 12.5ms
```

### TensorRT INT8

```
FPS (wall)    : 47.4
Preprocess    : 6.5ms
Inference     : 3.7ms
Postprocess   : 3.5ms
Total / frame : 13.7ms
```

---

## Key Observations

**Python preprocessing bottleneck (5.6ms):**
PIL/NumPy letterbox + normalize under Python GIL.
C++ equivalent: 1.7ms (3.3× faster).

**TRT inference is hardware-limited (same for Python and C++):**
```
Python FP16 TRT : 4.2ms
Python INT8 TRT : 3.5ms
C++ FP16 TRT    : ~4.2ms (estimated)
C++ INT8 TRT    : 3.5ms  ← confirmed identical
```

**Python INT8 postprocess regression:**
Ultralytics handles INT8 output tensor with additional processing
steps vs FP16. Postprocess: 3.2ms (INT8) vs 1.2ms (FP16).
Not present in C++ custom decoder.
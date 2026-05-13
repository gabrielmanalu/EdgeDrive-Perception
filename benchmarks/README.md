# EdgeDrive Perception — Benchmark Results

All benchmarks run on **Jetson Orin Nano Super 8GB** with `sudo jetson_clocks`
(max clocks locked). Images preloaded into RAM to isolate compute from disk I/O.

---

## Summary

| Format | FPS | Preprocess | TRT/Infer | Postprocess | Total/frame | Power |
|---|---|---|---|---|---|---|
| Python FP32 | 29.1 | 5.2ms | 27.6ms | 1.1ms | 34.0ms | — |
| Python TRT FP16 | 87.6 | 5.6ms | 4.2ms | 1.2ms | 11.0ms | — |
| Python TRT INT8 | 78.5 | 5.6ms | 3.5ms | 3.2ms | 12.3ms | — |
| **C++ TRT INT8** | **193.3** | **1.7ms** | **3.5ms** | — | **5.1ms** | **~12.2W** |

C++ pipeline is **6.6× faster** than Python FP32 end-to-end.
TRT-only time matches between Python (3.5ms) and C++ (3.5ms) — same engine, same GPU.

---

## Detailed Results

- [`results/jetson_python.md`](results/jetson_python.md) — Python FP32 / FP16 / INT8 full breakdown
- [`results/jetson_cpp.md`](results/jetson_cpp.md) — C++ TensorRT INT8 full breakdown
- [`results/quantization_colab.md`](results/quantization_colab.md) — TFLite quantization accuracy (Colab)

---

## Benchmark Conditions

```
Hardware  : Jetson Orin Nano Super 8GB Developer Kit (67 TOPS)
JetPack   : R36.4.7
CUDA      : 12.6
TensorRT  : 10.3.0
PyTorch   : 2.8.0
OpenCV    : 4.8.0
Clocks    : sudo jetson_clocks (CPU 1728 MHz, GPU 1017 MHz)
Dataset   : 404 nuScenes Mini CAM_FRONT images
Preloaded : All images loaded into RAM before timing starts
Duration  : 60s sustained run (single image: 20 iterations)
```
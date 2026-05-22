/*
 * lidar-backbone.cu — EdgeDrive Perception: 3-Level FPN Backbone
 * ===============================================================
 *
 * Updated from single-level to 3-level FPN for full nuScenes detection.
 *
 * ONNX: pointpillars_nuscenes_fpn3.onnx (exported with all 3 FPN levels)
 *
 * TRT engine bindings (10 tensors):
 *   [0] pseudo_image : input  [1, 64, 400, 400] float32
 *   [1] cls_l0       : output [1, 80, 200, 200] float32  small objects (ped/bike)
 *   [2] box_l0       : output [1, 72, 200, 200] float32
 *   [3] dir_l0       : output [1, 16, 200, 200] float32
 *   [4] cls_l1       : output [1, 80, 100, 100] float32  medium objects (car)
 *   [5] box_l1       : output [1, 72, 100, 100] float32
 *   [6] dir_l1       : output [1, 16, 100, 100] float32
 *   [7] cls_l2       : output [1, 80,  50,  50] float32  large objects (truck/bus)
 *   [8] box_l2       : output [1, 72,  50,  50] float32
 *   [9] dir_l2       : output [1, 16,  50,  50] float32
 *
 * Anchor scales per level (AlignedAnchor3DRangeGenerator scales=[1,2,4]):
 *   Level 0 (scale=1): anchors × 1.0  [2.60, 1.73, 1.00, 0.40]
 *   Level 1 (scale=2): anchors × 2.0  [5.20, 3.46, 2.00, 0.80]
 *   Level 2 (scale=4): anchors × 4.0  [10.40, 6.93, 4.00, 1.60]
 *
 * CHW scatter kernel unchanged — produces [1, 64, 400, 400] pseudo-image.
 */

#include <cuda_fp16.h>
#include <numeric>
#include "lidar-backbone.hpp"
#include "common/check.hpp"
#include "common/launch.cuh"
#include "common/tensorrt.hpp"

// ── CHW Scatter kernel ────────────────────────────────────────────────────────

static const int PP_BLOCK    = 64;
static const int PP_FEATURES = 64;
static const int GRID_X      = 400;
static const int GRID_Y      = 400;

__global__ void pillarScatterCHW_kernel(
    const half*          pillar_features,
    const unsigned int*  coords,
    const unsigned int*  params,
    unsigned int         grid_x,
    unsigned int         grid_y,
    float*               pseudo_image)
{
    int pillar_idx = blockIdx.x * PP_BLOCK + threadIdx.x;
    const int num_pillars = params[0];
    if (pillar_idx >= num_pillars) return;

    uint4 coord = ((const uint4*)coords)[pillar_idx];
    unsigned int x = coord.w;
    unsigned int y = coord.z;
    if (x >= grid_x || y >= grid_y) return;

    for (int c = 0; c < PP_FEATURES; c++) {
        float val = __half2float(
            pillar_features[pillar_idx * PP_FEATURES + c]);
        pseudo_image[c * grid_y * grid_x + y * grid_x + x] = val;
    }
}

// ── Backbone ──────────────────────────────────────────────────────────────────

namespace pointpillar {
namespace lidar {

class BackboneImplement : public Backbone {
public:
    virtual ~BackboneImplement() {
        if (pseudo_image_) checkRuntime(cudaFree(pseudo_image_));
        for (int i = 0; i < 3; i++) {
            if (cls_[i]) checkRuntime(cudaFree(cls_[i]));
            if (box_[i]) checkRuntime(cudaFree(box_[i]));
            if (dir_[i]) checkRuntime(cudaFree(dir_[i]));
        }
    }

    bool init(const std::string& model) {
        engine_ = TensorRT::load(model);
        if (engine_ == nullptr) return false;

        // Verify binding count
        // Single-level: 4 bindings  (legacy)
        // Three-level:  10 bindings (new)
        int nb = engine_->num_bindings();
        is_fpn3_ = (nb >= 10);

        printf("[Backbone] %s mode (%d bindings)\n",
               is_fpn3_ ? "FPN3" : "SingleLevel", nb);

        // Allocate pseudo-image
        checkRuntime(cudaMalloc(&pseudo_image_,
            sizeof(float) * PP_FEATURES * GRID_Y * GRID_X));

        if (is_fpn3_) {
            // 3-level: bindings 1-3, 4-6, 7-9
            for (int lvl = 0; lvl < 3; lvl++) {
                cls_dims_[lvl] = engine_->static_dims(1 + lvl * 3);
                box_dims_[lvl] = engine_->static_dims(2 + lvl * 3);
                dir_dims_[lvl] = engine_->static_dims(3 + lvl * 3);
                alloc_buf(cls_[lvl], cls_dims_[lvl]);
                alloc_buf(box_[lvl], box_dims_[lvl]);
                alloc_buf(dir_[lvl], dir_dims_[lvl]);
            }
        } else {
            // Single-level fallback
            cls_dims_[0] = engine_->static_dims(1);
            box_dims_[0] = engine_->static_dims(2);
            dir_dims_[0] = engine_->static_dims(3);
            alloc_buf(cls_[0], cls_dims_[0]);
            alloc_buf(box_[0], box_dims_[0]);
            alloc_buf(dir_[0], dir_dims_[0]);
        }
        return true;
    }

    void alloc_buf(float*& ptr, const std::vector<int>& dims) {
        int32_t vol = std::accumulate(dims.begin(), dims.end(), 1,
                                      std::multiplies<int32_t>());
        checkRuntime(cudaMalloc(&ptr, vol * sizeof(float)));
    }

    virtual void print() override {
        engine_->print(is_fpn3_ ? "Lidar Backbone FPN3 (nuScenes)"
                                 : "Lidar Backbone Single (nuScenes)");
    }

    virtual void forward(const nvtype::half* voxels,
                         const unsigned int* voxel_idxs,
                         const unsigned int* params,
                         void* stream = nullptr) override
    {
        cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);

        // Step 1: CHW Scatter → pseudo-image
        checkRuntime(cudaMemsetAsync(pseudo_image_, 0,
            sizeof(float) * PP_FEATURES * GRID_Y * GRID_X, _stream));

        unsigned int num_pillars = 0;
        checkRuntime(cudaMemcpyAsync(&num_pillars, params,
            sizeof(unsigned int), cudaMemcpyDeviceToHost, _stream));
        checkRuntime(cudaStreamSynchronize(_stream));

        if (num_pillars > 0) {
            int blocks = (num_pillars + PP_BLOCK - 1) / PP_BLOCK;
            pillarScatterCHW_kernel<<<blocks, PP_BLOCK, 0, _stream>>>(
                reinterpret_cast<const half*>(voxels),
                voxel_idxs, params,
                GRID_X, GRID_Y, pseudo_image_);
        }

        // Step 2: TRT forward
        if (is_fpn3_) {
            engine_->forward(
                {pseudo_image_,
                 cls_[0], box_[0], dir_[0],
                 cls_[1], box_[1], dir_[1],
                 cls_[2], box_[2], dir_[2]},
                static_cast<cudaStream_t>(_stream));
        } else {
            engine_->forward(
                {pseudo_image_, cls_[0], box_[0], dir_[0]},
                static_cast<cudaStream_t>(_stream));
        }
    }

    // Level 0 accessors (required by Backbone interface)
    virtual float* cls() override { return cls_[0]; }
    virtual float* box() override { return box_[0]; }
    virtual float* dir() override { return dir_[0]; }

    // Level 1 accessors
    virtual float* cls1() override { return cls_[1]; }
    virtual float* box1() override { return box_[1]; }
    virtual float* dir1() override { return dir_[1]; }

    // Level 2 accessors
    virtual float* cls2() override { return cls_[2]; }
    virtual float* box2() override { return box_[2]; }
    virtual float* dir2() override { return dir_[2]; }

    virtual bool is_fpn3() const override { return is_fpn3_; }

private:
    std::shared_ptr<TensorRT::Engine> engine_;
    float* pseudo_image_ = nullptr;
    float* cls_[3] = {nullptr, nullptr, nullptr};
    float* box_[3] = {nullptr, nullptr, nullptr};
    float* dir_[3] = {nullptr, nullptr, nullptr};
    std::vector<int> cls_dims_[3], box_dims_[3], dir_dims_[3];
    bool is_fpn3_ = false;
};

std::shared_ptr<Backbone> create_backbone(const std::string& model) {
    std::shared_ptr<BackboneImplement> instance(new BackboneImplement());
    if (!instance->init(model)) instance.reset();
    return instance;
}

};  // namespace lidar
};  // namespace pointpillar
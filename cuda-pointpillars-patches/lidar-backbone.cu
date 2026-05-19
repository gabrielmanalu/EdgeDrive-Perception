/*
 * lidar-backbone.cu — EdgeDrive Perception: nuScenes Backbone with CHW Scatter
 * ==============================================================================
 *
 * Patched from NVIDIA-AI-IOT/CUDA-PointPillars for our split-pipeline approach.
 *
 * Key architectural change:
 *
 *   ORIGINAL (KITTI):
 *     TRT engine contains VFE + PPScatterPlugin + SECOND + FPN + Head (6 bindings)
 *     forward() passes: {voxels, coords, params, cls, box, dir}
 *     Scatter from HWC8 → 2D BEV grid happens INSIDE TRT via custom plugin
 *
 *   OURS (nuScenes):
 *     TRT engine contains SECOND + FPN + Anchor3DHead only (4 bindings)
 *     forward() passes: {pseudo_image, cls, box, dir}
 *     Scatter happens OUTSIDE TRT via pillarScatterCHW_kernel (this file)
 *     VFE still runs in lidar-voxelization.cu as before
 *
 * Why the split?
 *   Our model was exported from mmdetection3d (MVXFasterRCNN) as backbone+neck+head
 *   only. The HardVFE and PointPillarsScatter are not part of the exported ONNX.
 *   Rather than re-export with a custom PPScatterPlugin, we implement the scatter
 *   as a lightweight CUDA kernel that outputs CHW format directly.
 *
 * pillarScatterCHW_kernel:
 *   Converts sparse pillar features [N, 64] → dense pseudo-image [1, 64, 400, 400]
 *   Output format: standard PyTorch CHW (channel-first), float32
 *   Input format:  pillar features in half precision from lidar-voxelization.cu
 *
 *   Grid params (nuScenes):
 *     GRID_X = GRID_Y = 400  (point_cloud_range ±50m / voxel_size 0.25m)
 *     PP_FEATURES = 64       (HardVFE output channels)
 *
 * TRT engine bindings (4 tensors):
 *   [0] pseudo_image : input  [1, 64, 400, 400] float32
 *   [1] cls_scores   : output [1, 80, 200, 200] float32  (10 classes × 8 anchors)
 *   [2] bbox_preds   : output [1, 72, 200, 200] float32  (9 values   × 8 anchors)
 *   [3] dir_preds    : output [1, 16, 200, 200] float32  (2 dirs     × 8 anchors)
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
        if (cls_)          checkRuntime(cudaFree(cls_));
        if (box_)          checkRuntime(cudaFree(box_));
        if (dir_)          checkRuntime(cudaFree(dir_));
    }

    bool init(const std::string& model) {
        engine_ = TensorRT::load(model);
        if (engine_ == nullptr) return false;

        // Our engine: [0]=pseudo_image, [1]=cls, [2]=box, [3]=dir
        cls_dims_ = engine_->static_dims(1);
        box_dims_ = engine_->static_dims(2);
        dir_dims_ = engine_->static_dims(3);

        // Allocate pseudo-image [1, 64, 400, 400]
        checkRuntime(cudaMalloc(&pseudo_image_,
            sizeof(float) * PP_FEATURES * GRID_Y * GRID_X));

        int32_t vol;
        vol = std::accumulate(cls_dims_.begin(), cls_dims_.end(), 1,
                              std::multiplies<int32_t>());
        checkRuntime(cudaMalloc(&cls_, vol * sizeof(float)));

        vol = std::accumulate(box_dims_.begin(), box_dims_.end(), 1,
                              std::multiplies<int32_t>());
        checkRuntime(cudaMalloc(&box_, vol * sizeof(float)));

        vol = std::accumulate(dir_dims_.begin(), dir_dims_.end(), 1,
                              std::multiplies<int32_t>());
        checkRuntime(cudaMalloc(&dir_, vol * sizeof(float)));

        return true;
    }

    virtual void print() override { engine_->print("Lidar Backbone (nuScenes)"); }

    virtual void forward(const nvtype::half* voxels,
                         const unsigned int* voxel_idxs,
                         const unsigned int* params,
                         void* stream = nullptr) override
    {
        cudaStream_t _stream = reinterpret_cast<cudaStream_t>(stream);

        // Step 1: Scatter pillar features → pseudo-image [1, 64, 400, 400] CHW
        // Zero pseudo-image (sparse scatter — empty pillars stay 0)
        checkRuntime(cudaMemsetAsync(pseudo_image_, 0,
            sizeof(float) * PP_FEATURES * GRID_Y * GRID_X, _stream));

        // Get num_pillars from device params
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

        // Step 2: Run backbone TRT engine
        // Inputs:  [pseudo_image]
        // Outputs: [cls_, box_, dir_]
        engine_->forward(
            {pseudo_image_, cls_, box_, dir_},
            static_cast<cudaStream_t>(_stream));
    }

    virtual float* cls() override { return cls_; }
    virtual float* box() override { return box_; }
    virtual float* dir() override { return dir_; }

private:
    std::shared_ptr<TensorRT::Engine> engine_;
    float* pseudo_image_ = nullptr;
    float* cls_          = nullptr;
    float* box_          = nullptr;
    float* dir_          = nullptr;
    std::vector<int> cls_dims_, box_dims_, dir_dims_;
};

std::shared_ptr<Backbone> create_backbone(const std::string& model) {
    std::shared_ptr<BackboneImplement> instance(new BackboneImplement());
    if (!instance->init(model)) instance.reset();
    return instance;
}

};  // namespace lidar
};  // namespace pointpillar

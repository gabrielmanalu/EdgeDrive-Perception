/*
 * pointpillar.cpp — EdgeDrive Perception: 3-Level FPN Core
 * ==========================================================
 *
 * Patched from NVIDIA-AI-IOT/CUDA-PointPillars.
 * Updated to run postprocess on all 3 FPN levels and merge results.
 *
 * Pipeline per frame:
 *   1. Voxelization (C++ CUDA kernel)
 *   2. CHW scatter → pseudo-image [1, 64, 400, 400]
 *   3. TRT backbone FPN3 → 9 output tensors (3 levels × cls/box/dir)
 *   4. Postprocess level 0 (200×200, scale=1) → small object detections
 *   5. Postprocess level 1 (100×100, scale=2) → medium object detections
 *   6. Postprocess level 2 ( 50× 50, scale=4) → large object detections
 *   7. Merge all level results
 *
 * Per-level anchor scaling (AlignedAnchor3DRangeGenerator scales=[1,2,4]):
 *   Level 0 (scale=1): anchors as-is  → pedestrian, bicycle, cone, barrier
 *   Level 1 (scale=2): anchors × 2.0  → car, construction_vehicle
 *   Level 2 (scale=4): anchors × 4.0  → truck, bus, trailer
 *
 * NMS pre-filter: top-1000 per level before NMS (nuScenes test_cfg nms_pre=1000)
 *
 * Performance (Jetson Orin Nano Super, 25W mode):
 *   Voxelization  : ~0.8ms
 *   Backbone FPN3 : ~18-54ms (warm: ~18ms)
 *   Decoder+NMS   : ~18-21ms (3 levels)
 *   Total         : ~39-75ms  (~15-25 FPS)
 *   Detections    : 140-155 per frame on nuScenes (43K points)
 */

#include "pointpillar.hpp"
#include <numeric>
#include "common/check.hpp"
#include "common/timer.hpp"
#include "common/tensor.hpp"

namespace pointpillar {
namespace lidar {

// Build PostProcessParameter for a given FPN level
static PostProcessParameter make_level_param(
    const PostProcessParameter& base, int level)
{
    PostProcessParameter pp = base;
    float scale = (level == 0) ? 1.0f : (level == 1) ? 2.0f : 4.0f;

    // Feature size halves each level
    pp.feature_size = nvtype::Int2(
        base.feature_size.x >> level,
        base.feature_size.y >> level);

    // Scale anchor w, l, h (not rotation)
    for (int i = 0; i < pp.num_anchors; i++) {
        float* a = pp.anchors + i * pp.len_per_anchor;
        a[0] *= scale;  // w
        a[1] *= scale;  // l
        a[2] *= scale;  // h
    }
    return pp;
}

class CoreImplement : public Core {
public:
    virtual ~CoreImplement() {
        if (lidar_points_device_) checkRuntime(cudaFree(lidar_points_device_));
        if (lidar_points_host_)   checkRuntime(cudaFreeHost(lidar_points_host_));
    }

    bool init(const CoreParameter& param) {
        lidar_voxelization_ = create_voxelization(param.voxelization);
        if (!lidar_voxelization_) { printf("Failed voxelization\n"); return false; }

        lidar_backbone_ = create_backbone(param.lidar_model);
        if (!lidar_backbone_) { printf("Failed backbone\n"); return false; }

        // Create PostProcess instances for each level.
        // max_levels (CoreParameter, default 3) selects how many FPN levels
        // to decode. Both standalone and the ROS2 node use the default of 3
        // (full FPN3) unless a caller explicitly lowers it. The ': 1' branch
        // only applies to a non-FPN3 (single-level) engine; the FPN3 plan
        // used here reports is_fpn3()==true, so 'levels' = param.max_levels.
        int levels = lidar_backbone_->is_fpn3()
                     ? param.max_levels : 1;
        if (levels < 1) levels = 1;
        if (levels > 3) levels = 3;
        for (int lvl = 0; lvl < levels; lvl++) {
            auto pp_param = make_level_param(param.lidar_post, lvl);
            lidar_postprocess_[lvl] = create_postprocess(pp_param);
            if (!lidar_postprocess_[lvl]) {
                printf("Failed postprocess level %d\n", lvl); return false;
            }
        }
        num_levels_ = levels;

        capacity_points_ = 300000;
        bytes_capacity_points_ =
            capacity_points_ * param.voxelization.num_feature * sizeof(float);
        checkRuntime(cudaMalloc(&lidar_points_device_, bytes_capacity_points_));
        checkRuntime(cudaMallocHost(&lidar_points_host_, bytes_capacity_points_));
        param_ = param;
        return true;
    }

    std::vector<BoundingBox> run_inference(
        const float* lidar_points, int num_points,
        void* stream, bool timer)
    {
        int cappoints = static_cast<int>(capacity_points_);
        num_points = std::min(cappoints, num_points);

        cudaStream_t _stream = static_cast<cudaStream_t>(stream);
        std::vector<float> times;

        if (timer) {
            timer_.start(_stream);
            printf("==================PointPillars FPN%d==================\n",
                   num_levels_);
        }

        size_t bytes_pts = num_points * param_.voxelization.num_feature * sizeof(float);
        checkRuntime(cudaMemcpyAsync(lidar_points_host_, lidar_points, bytes_pts,
                                    cudaMemcpyHostToHost, _stream));
        checkRuntime(cudaMemcpyAsync(lidar_points_device_, lidar_points_host_,
                                    bytes_pts, cudaMemcpyHostToDevice, _stream));
        if (timer) timer_.stop("[NoSt] CopyLidar");

        if (timer) timer_.start(_stream);
        lidar_voxelization_->forward(lidar_points_device_, num_points, _stream);
        if (timer) times.emplace_back(timer_.stop("Lidar Voxelization"));

        if (timer) timer_.start(_stream);
        lidar_backbone_->forward(
            lidar_voxelization_->features(),
            lidar_voxelization_->coords(),
            lidar_voxelization_->params(), _stream);
        if (timer) times.emplace_back(timer_.stop("Lidar Backbone"));

        if (timer) timer_.start(_stream);

        // Run postprocess on all available levels
        float* cls_ptrs[3] = {
            lidar_backbone_->cls(),
            lidar_backbone_->cls1(),
            lidar_backbone_->cls2()
        };
        float* box_ptrs[3] = {
            lidar_backbone_->box(),
            lidar_backbone_->box1(),
            lidar_backbone_->box2()
        };
        float* dir_ptrs[3] = {
            lidar_backbone_->dir(),
            lidar_backbone_->dir1(),
            lidar_backbone_->dir2()
        };

        for (int lvl = 0; lvl < num_levels_; lvl++) {
            if (cls_ptrs[lvl] && box_ptrs[lvl] && dir_ptrs[lvl]) {
                lidar_postprocess_[lvl]->forward(
                    cls_ptrs[lvl], box_ptrs[lvl], dir_ptrs[lvl], _stream);
            }
        }

        // Merge all level detections
        std::vector<BoundingBox> result;
        for (int lvl = 0; lvl < num_levels_; lvl++) {
            if (lidar_postprocess_[lvl]) {
                auto dets = lidar_postprocess_[lvl]->bndBoxVec();
                result.insert(result.end(), dets.begin(), dets.end());
            }
        }

        if (timer) {
            times.emplace_back(timer_.stop("Lidar Decoder+NMS all levels"));
            float total = std::accumulate(times.begin(), times.end(), 0.0f,
                                          std::plus<float>());
            printf("Total: %.3f ms\n", total);
            printf("Detections after NMS: %zu\n", result.size());
            printf("====================================================\n");
        }

        return result;
    }

    virtual std::vector<BoundingBox> forward(
        const float* lidar_points, int num_points, void* stream) override {
        return run_inference(lidar_points, num_points, stream, enable_timer_);
    }

    virtual void set_timer(bool enable) override { enable_timer_ = enable; }
    virtual void print() override { lidar_backbone_->print(); }

private:
    CoreParameter param_;
    nv::EventTimer timer_;
    float* lidar_points_device_ = nullptr;
    float* lidar_points_host_   = nullptr;
    size_t capacity_points_     = 0;
    size_t bytes_capacity_points_ = 0;
    int    num_levels_          = 1;

    std::shared_ptr<Voxelization>  lidar_voxelization_;
    std::shared_ptr<Backbone>      lidar_backbone_;
    std::shared_ptr<PostProcess>   lidar_postprocess_[3] = {nullptr, nullptr, nullptr};
    bool enable_timer_ = false;
};

std::shared_ptr<Core> create_core(const CoreParameter& param) {
    std::shared_ptr<CoreImplement> instance(new CoreImplement());
    if (!instance->init(param)) instance.reset();
    return instance;
}

};  // namespace lidar
};  // namespace pointpillar
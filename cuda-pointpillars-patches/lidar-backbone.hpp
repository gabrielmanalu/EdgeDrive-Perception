/*
 * lidar-backbone.hpp — EdgeDrive Perception: FPN3 Backbone Interface
 * ====================================================================
 *
 * Patched from NVIDIA-AI-IOT/CUDA-PointPillars for nuScenes FPN3 model.
 *
 * Key changes from original:
 *
 *   Multi-level virtual accessors added to Backbone base class:
 *     Original interface only exposed level 0:
 *       virtual float* cls() = 0;
 *       virtual float* box() = 0;
 *       virtual float* dir() = 0;
 *
 *     Extended with default-nullptr FPN level 1 and 2 accessors:
 *       virtual float* cls1() { return nullptr; }  // 100×100
 *       virtual float* box1() { return nullptr; }
 *       virtual float* dir1() { return nullptr; }
 *       virtual float* cls2() { return nullptr; }  //  50×50
 *       virtual float* box2() { return nullptr; }
 *       virtual float* dir2() { return nullptr; }
 *       virtual bool is_fpn3() const { return false; }
 *
 *   Backwards compatible: single-level engines return nullptr for
 *   levels 1 and 2. pointpillar.cpp checks non-null before running
 *   additional postprocess passes.
 *
 *   FPN3 engine binding layout (pointpillar_fpn3.plan):
 *     Input  [0]: pseudo_image {1, 64, 400, 400} float32
 *     Output [1-3]:  cls_l0, box_l0, dir_l0  {1,80/72/16,200,200}
 *     Output [4-6]:  cls_l1, box_l1, dir_l1  {1,80/72/16,100,100}
 *     Output [7-9]:  cls_l2, box_l2, dir_l2  {1,80/72/16, 50, 50}
 *
 *   Engine exported from:
 *     pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py (mmdetection3d)
 *     Checkpoint: pointpillars_nuscenes.pth
 *     ONNX: pointpillars_nuscenes_fpn3.onnx
 *     Built with: trtexec --fp16 --plugins=libpointpillar_core.so
 */

#ifndef __LIDAR_BACKBONE_HPP__
#define __LIDAR_BACKBONE_HPP__
#include <memory>
#include <string>
#include <vector>
#include "common/dtype.hpp"
namespace pointpillar {
namespace lidar {
class Backbone {
 public:
    virtual void forward(const nvtype::half* voxels, const unsigned int* voxel_idxs, const unsigned int* params, void* stream = nullptr) = 0;

    // Level 0 (always available — small objects, 200×200)
    virtual float* cls() = 0;
    virtual float* box() = 0;
    virtual float* dir() = 0;

    // Levels 1 and 2 (FPN3 model only — default nullptr for single-level)
    virtual float* cls1() { return nullptr; }
    virtual float* box1() { return nullptr; }
    virtual float* dir1() { return nullptr; }
    virtual float* cls2() { return nullptr; }
    virtual float* box2() { return nullptr; }
    virtual float* dir2() { return nullptr; }

    virtual bool is_fpn3() const { return false; }
    virtual void print() = 0;
};
std::shared_ptr<Backbone> create_backbone(const std::string& model);
};  // namespace lidar
};  // namespace pointpillar
#endif  // __LIDAR_BACKBONE_HPP__
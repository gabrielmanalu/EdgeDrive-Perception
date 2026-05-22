/*
 * pointpillar.hpp — EdgeDrive Perception: PointPillars Core Interface
 * ====================================================================
 *
 * Patched from NVIDIA-AI-IOT/CUDA-PointPillars for nuScenes dataset.
 *
 * Key changes from original KITTI version:
 *
 *   VoxelizationParameter:
 *     min/max range   : KITTI [0,-40,-3, 70.4,40,1]
 *                     → nuScenes [-50,-50,-5, 50,50,3]
 *     voxel_size      : [0.16,0.16,4.0] → [0.25,0.25,8.0]
 *     grid_size       : KITTI 432×496 → nuScenes 400×400
 *     num_feature     : 4 (ring channel stripped from nuScenes 5-field PCD)
 *     max_voxels      : 30000 (sufficient for nuScenes 43K pts at 0.25m)
 *
 *   PostProcessParameter:
 *     num_classes     : 3  → 10  (full nuScenes 10-class set)
 *     num_anchors     : 6  → 8   (4 anchor sizes × 2 rotations)
 *     len_per_anchor  : 4  (w, l, h, rotation)
 *     anchors[]       : nuScenes 4-size anchor grid
 *                       [1.6,3.9,1.56] car
 *                       [0.6,1.76,1.73] pedestrian
 *                       [0.6,1.6,1.28]  bicycle/motorcycle
 *                       [2.5,6.93,2.73] bus/truck/trailer
 *     bbox_code_size  : 7 → 9  (DeltaXYZWLHRBBoxCoder with vel)
 *     score_thresh    : 0.1 → 0.05
 *     nms_thresh      : 0.01 → 0.2
 *     dir_offset      : 0.785 → -0.785
 *     dir_limit_offset: 0.0 → 0.0
 *     feature_size    : nvtype::Int2(200, 200)  (level 0, 400×400/2)
 *
 *   CoreParameter:
 *     Added max_levels field (default 3):
 *       1 = level 0 only (pedestrian/bicycle, memory-safe for ROS2)
 *       2 = level 0+1 (+ cars)
 *       3 = all levels (+ trucks/buses, standalone only)
 *     lidar_model: "../model/pointpillar_fpn3.plan"
 *
 *   nuScenes classes (confirmed via mmdetection3d inference_detector):
 *     0:car  1:truck  2:construction_vehicle  3:bus  4:trailer
 *     5:barrier  6:motorcycle  7:pedestrian  8:bicycle  9:traffic_cone
 */

#ifndef __POINTPILLAR_HPP__
#define __POINTPILLAR_HPP__

#include "lidar-voxelization.hpp"
#include "lidar-backbone.hpp"
#include "lidar-postprocess.hpp"
// #include "nms.hpp"

namespace pointpillar {
namespace lidar {

struct CoreParameter {
    int max_levels = 3;  // FPN levels to use (1=level0 only, 3=full FPN3)
    VoxelizationParameter voxelization;
    std::string lidar_model;
    PostProcessParameter lidar_post;
};

class Core {
    public:
        virtual std::vector<BoundingBox> forward(const float *lidar_points, int num_points, void *stream) = 0;

        virtual void print() = 0;
        virtual void set_timer(bool enable) = 0;
};

std::shared_ptr<Core> create_core(const CoreParameter &param);

};  // namespace lidar
};  // namespace pointpillar

#endif  // __POINTPILLAR_HPP__

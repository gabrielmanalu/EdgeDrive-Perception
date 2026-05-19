/*
 * lidar-postprocess.hpp — EdgeDrive Perception: nuScenes PostProcess Parameters
 * ==============================================================================
 *
 * Patched from NVIDIA-AI-IOT/CUDA-PointPillars for nuScenes dataset.
 *
 * Key changes from KITTI defaults:
 *
 *   num_classes  : 3  → 10  (nuScenes 10-class detection)
 *   num_anchors  : 6  → 8   (4 anchor sizes × 2 rotations)
 *   anchors[]    : KITTI car/ped/cyclist → nuScenes sizes from mmdetection3d config
 *   num_box_values: 7 → 9   (DeltaXYZWLHRBBoxCoder code_size=9)
 *   score_thresh : 0.1 → 0.05 (nuScenes uses lower threshold)
 *   nms_thresh   : 0.01 → 0.2  (nuScenes NMS threshold)
 *   dir_offset   : 0.785 → -0.785 (from mmdetection3d anchor3d_head config)
 *
 * nuScenes classes (10):
 *   0: car               5: motorcycle
 *   1: truck             6: bicycle
 *   2: bus               7: traffic_cone
 *   3: trailer           8: barrier
 *   4: construction_veh  9: (unused/background)
 *
 * Anchors (8 per grid location, from mmdetection3d AlignedAnchor3DRangeGenerator):
 *   Anchor format: [w, l, h, rotation]
 *   [2.60, 0.87, 1.0, 0.0]    car rot=0°
 *   [2.60, 0.87, 1.0, 1.5708] car rot=90°
 *   [1.73, 0.58, 1.0, 0.0]    pedestrian rot=0°
 *   [1.73, 0.58, 1.0, 1.5708] pedestrian rot=90°
 *   [1.00, 1.00, 1.0, 0.0]    bicycle rot=0°
 *   [1.00, 1.00, 1.0, 1.5708] bicycle rot=90°
 *   [0.40, 0.40, 1.0, 0.0]    cone/barrier rot=0°
 *   [0.40, 0.40, 1.0, 1.5708] cone/barrier rot=90°
 *
 * Model output dimensions (matching our exported ONNX):
 *   cls_scores : [1, 80,  200, 200]  (10 classes × 8 anchors)
 *   bbox_preds : [1, 72,  200, 200]  (9 values   × 8 anchors)
 *   dir_preds  : [1, 16,  200, 200]  (2 dirs     × 8 anchors)
 */

#ifndef __LIDAR_POSTPROCESS_HPP__
#define __LIDAR_POSTPROCESS_HPP__

#include <memory>
#include <vector>
#include "common/dtype.hpp"

namespace pointpillar {
namespace lidar {

struct BoundingBox {
    float x;
    float y;
    float z;
    float w;
    float l;
    float h;
    float rt;
    int id;
    float score;
    BoundingBox(){};
    BoundingBox(float x_, float y_, float z_, float w_, float l_, float h_, float rt_, int id_, float score_)
        : x(x_), y(y_), z(z_), w(w_), l(l_), h(h_), rt(rt_), id(id_), score(score_) {}
};

struct PostProcessParameter {
    nvtype::Float3 min_range;
    nvtype::Float3 max_range;
    nvtype::Int2 feature_size;
    int num_classes = 3;
    int num_anchors = 6;
    int len_per_anchor = 4;
    float anchors[32] = {
            3.9,1.6,1.56,0.0,
            3.9,1.6,1.56,1.57,
            0.8,0.6,1.73,0.0,
            0.8,0.6,1.73,1.57,
            1.76,0.6,1.73,0.0,
            1.76,0.6,1.73,1.57,
        };
    nvtype::Float3 anchor_bottom_heights{-1.78,-0.6,-0.6};
    int num_box_values = 9;
    float score_thresh = 0.1;
    float dir_offset = 0.78539;
    float nms_thresh = 0.01;
};

class PostProcess {
    public:
        virtual void forward(const float* cls, const float* box, const float* dir, void* stream) = 0;

        virtual std::vector<BoundingBox> bndBoxVec() = 0;
};

std::shared_ptr<PostProcess> create_postprocess(const PostProcessParameter& param);

};  // namespace lidar
};  // namespace pointpillar

#endif  // __LIDAR_POSTPROCESS_HPP__

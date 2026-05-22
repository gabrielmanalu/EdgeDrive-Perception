/*
 * main.cpp — EdgeDrive Perception: nuScenes PointPillars Configuration
 * =====================================================================
 *
 * Patched from NVIDIA-AI-IOT/CUDA-PointPillars for nuScenes dataset.
 *
 * Key changes from KITTI defaults:
 *
 *   VOXELIZATION:
 *     KITTI : range [0,-40,-3, 70.4,40,1], voxel [0.16,0.16,4.0]
 *     Ours  : range [-50,-50,-5, 50,50,3], voxel [0.25,0.25,8.0]
 *             (nuScenes full 100m × 100m coverage at 0.25m resolution)
 *     Note  : nuScenes LiDAR has 5 fields (x,y,z,intensity,ring)
 *             but num_feature=4 — ring channel stripped in lidar_detection_node
 *
 *   ENGINE:
 *     KITTI : ../model/pointpillar.plan  (KITTI-trained, 6 bindings)
 *     Ours  : ../model/pointpillar_fpn3.plan (nuScenes FPN3, 10 bindings)
 *             Built from pointpillars_nuscenes_fpn3.onnx (mmdetection3d export)
 *
 *   POSTPROCESS:
 *     KITTI : 3 classes, 6 anchors, bbox_code=7, feature_size from KITTI grid
 *     Ours  : 10 classes, 8 anchors, bbox_code=9, feature_size 200×200 (level 0)
 *             Levels 1 (100×100) and 2 (50×50) handled by make_level_param()
 *             in pointpillar.cpp which scales anchor sizes per FPN level
 *
 * ONNX export:
 *   Config     : pointpillars_hv_fpn_sbn-all_8xb4-2x_nus-3d.py (mmdetection3d)
 *   Checkpoint : pointpillars_nuscenes.pth
 *   ONNX       : pointpillars_nuscenes_fpn3.onnx (all 3 FPN levels, 9 outputs)
 *   Engine     : pointpillar_fpn3.plan (TRT FP16, SM87)
 *
 * Performance (Jetson Orin Nano Super, 25W, nuScenes data 43K pts):
 *   Warm inference : ~39ms total (~25 FPS)
 *   Cold start     : ~75ms (first frame TRT warmup)
 *   Detections     : 140-155 per frame (10 classes detected)
 */

#include <cuda_runtime.h>

#include <string.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <dirent.h>

#include "pointpillar.hpp"
#include "common/check.hpp"

void GetDeviceInfo(void)
{
  cudaDeviceProp prop;

  int count = 0;
  cudaGetDeviceCount(&count);
  printf("\nGPU has cuda devices: %d\n", count);
  for (int i = 0; i < count; ++i) {
    cudaGetDeviceProperties(&prop, i);
    printf("----device id: %d info----\n", i);
    printf("  GPU : %s \n", prop.name);
    printf("  Capbility: %d.%d\n", prop.major, prop.minor);
    printf("  Global memory: %luMB\n", prop.totalGlobalMem >> 20);
    printf("  Const memory: %luKB\n", prop.totalConstMem  >> 10);
    printf("  SM in a block: %luKB\n", prop.sharedMemPerBlock >> 10);
    printf("  warp size: %d\n", prop.warpSize);
    printf("  threads in a block: %d\n", prop.maxThreadsPerBlock);
    printf("  block dim: (%d,%d,%d)\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("  grid dim: (%d,%d,%d)\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
  }
  printf("\n");
}

bool hasEnding(std::string const &fullString, std::string const &ending)
{
    if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare (fullString.length() - ending.length(), ending.length(), ending));
    } else {
        return false;
    }
}

int getFolderFile(const char *path, std::vector<std::string>& files, const char *suffix = ".bin")
{
    DIR *dir;
    struct dirent *ent;
    if ((dir = opendir(path)) != NULL) {
        while ((ent = readdir (dir)) != NULL) {
            std::string file = ent->d_name;
            if(hasEnding(file, suffix)){
                files.push_back(file.substr(0, file.length()-4));
            }
        }
        closedir(dir);
    } else {
        printf("No such folder: %s.", path);
        exit(EXIT_FAILURE);
    }
    return EXIT_SUCCESS;
}

int loadData(const char *file, void **data, unsigned int *length)
{
    std::fstream dataFile(file, std::ifstream::in);

    if (!dataFile.is_open()) {
        std::cout << "Can't open files: "<< file<<std::endl;
        return -1;
    }

    unsigned int len = 0;
    dataFile.seekg (0, dataFile.end);
    len = dataFile.tellg();
    dataFile.seekg (0, dataFile.beg);

    char *buffer = new char[len];
    if (buffer==NULL) {
        std::cout << "Can't malloc buffer."<<std::endl;
        dataFile.close();
        exit(EXIT_FAILURE);
    }

    dataFile.read(buffer, len);
    dataFile.close();

    *data = (void*)buffer;
    *length = len;
    return 0;  
}

void SaveBoxPred(std::vector<pointpillar::lidar::BoundingBox> boxes, std::string file_name)
{
    std::ofstream ofs;
    ofs.open(file_name, std::ios::out);
    if (ofs.is_open()) {
        for (const auto box : boxes) {
          ofs << box.x << " ";
          ofs << box.y << " ";
          ofs << box.z << " ";
          ofs << box.w << " ";
          ofs << box.l << " ";
          ofs << box.h << " ";
          ofs << box.rt << " ";
          ofs << box.id << " ";
          ofs << box.score << "\n";
        }
    }
    else {
      std::cerr << "Output file cannot be opened!" << std::endl;
    }
    ofs.close();
    std::cout << "Saved prediction in: " << file_name << std::endl;
    return;
};

std::shared_ptr<pointpillar::lidar::Core> create_core() {
    pointpillar::lidar::VoxelizationParameter vp;
    // nuScenes parameters
    vp.min_range = nvtype::Float3(-50.0f, -50.0f, -5.0f);
    vp.max_range = nvtype::Float3( 50.0f,  50.0f,  3.0f);
    vp.voxel_size = nvtype::Float3(0.25f, 0.25f, 8.0f);
    vp.grid_size =
        vp.compute_grid_size(vp.max_range, vp.min_range, vp.voxel_size);
    vp.max_voxels = 30000;
    vp.max_points_per_voxel = 32;
    vp.max_points = 300000;
    vp.num_feature = 4;  // x, y, z, intensity (ring channel dropped)

    pointpillar::lidar::PostProcessParameter pp;
    pp.min_range = vp.min_range;
    pp.max_range = vp.max_range;
    pp.feature_size = nvtype::Int2(vp.grid_size.x/2, vp.grid_size.y/2);

    // nuScenes: 10 classes, 8 anchors (4 sizes x 2 rotations)
    // Anchor format: [w, l, h, rotation]
    // From mmdet3d config: sizes [[2.60,0.87,1.0],[1.73,0.58,1.0],[1.0,1.0,1.0],[0.4,0.4,1.0]]
    pp.num_classes  = 10;
    pp.num_anchors  = 8;
    pp.len_per_anchor = 4;
    // anchors: w, l, h, rotation  (4 sizes x 2 rotations)
    float nuscenes_anchors[32] = {
        2.60f, 0.87f, 1.0f, 0.0f,      // car rot=0
        2.60f, 0.87f, 1.0f, 1.5708f,   // car rot=90
        1.73f, 0.58f, 1.0f, 0.0f,      // pedestrian rot=0
        1.73f, 0.58f, 1.0f, 1.5708f,   // pedestrian rot=90
        1.0f,  1.0f,  1.0f, 0.0f,      // bicycle rot=0
        1.0f,  1.0f,  1.0f, 1.5708f,   // bicycle rot=90
        0.4f,  0.4f,  1.0f, 0.0f,      // barrier/cone rot=0
        0.4f,  0.4f,  1.0f, 1.5708f,   // barrier/cone rot=90
    };
    memcpy(pp.anchors, nuscenes_anchors, sizeof(nuscenes_anchors));
    pp.anchor_bottom_heights = nvtype::Float3(-1.8f, -1.8f, -1.8f);
    pp.num_box_values = 9;   // DeltaXYZWLHRBBoxCoder code_size=9
    pp.score_thresh   = 0.05f;
    pp.nms_thresh     = 0.2f;
    pp.dir_offset     = -0.7854f;

    pointpillar::lidar::CoreParameter param;
    param.voxelization = vp;
    param.lidar_model = "../model/pointpillar_fpn3.plan";
    param.lidar_post = pp;
    return pointpillar::lidar::create_core(param);
}

static bool startswith(const char *s, const char *with, const char **last)
{
    while (*s++ == *with++)
    {
        if (*s == 0 || *with == 0)
            break;
    }
    if (*with == 0)
        *last = s + 1;
    return *with == 0;
}

static void help()
{
    printf(
        "Usage: \n"
        "    ./pointpillar in/ out/ --timer\n"
        "    Run pointpillar inference with .bin under in, save .text under out\n"
        "    Optional: --timer, enable timer log\n"
    );
    exit(EXIT_SUCCESS);
}

int main(int argc, char** argv) {

    if (argc < 3 || argc > 4)
        help();

    const char *in_dir  = argv[1];
    const char *out_dir  = argv[2];

    const char *value = nullptr;
    bool timer = false;

    if (argc == 4) {
        if (startswith(argv[3], "--timer", &value)) {
            timer = true;
        }
    }

    GetDeviceInfo();

    std::vector<std::string> files;
    getFolderFile(in_dir, files);
    std::cout << "Total " << files.size() << std::endl;

    auto core = create_core();
    if (core == nullptr) {
        printf("Core has been failed.\n");
        return -1;
    }

    cudaStream_t stream;
    cudaStreamCreate(&stream);
  
    core->print();
    core->set_timer(timer);

    for (const auto & file : files)
    {
        std::string dataFile = std::string(in_dir) + file + ".bin";

        std::cout << "\n<<<<<<<<<<<" <<std::endl;
        std::cout << "Load file: "<< dataFile <<std::endl;

        //load points cloud
        unsigned int length = 0;
        void *data = NULL;
        std::shared_ptr<char> buffer((char *)data, std::default_delete<char[]>());
        loadData(dataFile.data(), &data, &length);
        buffer.reset((char *)data);
        int points_size = length/sizeof(float)/4;
        std::cout << "Lidar points count: "<< points_size <<std::endl;
    
        auto bboxes = core->forward((float *)buffer.get(), points_size, stream);
        std::cout<<"Detections after NMS: "<< bboxes.size()<<std::endl;

        std::string save_file_name = std::string(out_dir) + file + ".txt";
        SaveBoxPred(bboxes, save_file_name);

        std::cout << ">>>>>>>>>>>" << std::endl;
    }

    checkRuntime(cudaStreamDestroy(stream));
    return 0;
}

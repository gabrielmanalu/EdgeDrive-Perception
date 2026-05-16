/**
 * bev_visualizer.cpp — Bird's Eye View visualization
 */

#include "bev_visualizer.hpp"
#include "yolo26_decoder.hpp"

#include <cmath>
#include <iomanip>
#include <sstream>

// ── Colors ────────────────────────────────────────────────────────────────────

static const cv::Scalar BG_COLOR    (0x0d, 0x11, 0x17);  // GitHub dark
static const cv::Scalar PANEL_COLOR (0x16, 0x1b, 0x22);  // panel
static const cv::Scalar ROAD_COLOR  (0x1e, 0x25, 0x2d);  // road surface
static const cv::Scalar GRID_COLOR  (0x2a, 0x35, 0x3f);  // grid ring
static const cv::Scalar GRID_TEXT   (0x58, 0x68, 0x78);  // distance labels
static const cv::Scalar EGO_FILL    (0x58, 0xa6, 0xff);  // blue
static const cv::Scalar EGO_ROOF    (0x7d, 0xbe, 0xff);  // lighter blue
static const cv::Scalar FOV_COLOR   (0x3f, 0xb9, 0x50);  // green cone
static const cv::Scalar LANE_COLOR  (0xd2, 0x99, 0x22);  // dashed yellow
static const cv::Scalar TITLE_COLOR (0x8b, 0x94, 0x9e);  // muted

// ── Constructor ───────────────────────────────────────────────────────────────

BEVVisualizer::BEVVisualizer(const cv::Mat& K, float cam_h)
    : cam_h_(cam_h)
{
    if (!K.empty() && K.rows == 3 && K.cols == 3) {
        K_  = K.clone();
        fx_ = static_cast<float>(K.at<double>(0, 0));
        fy_ = static_cast<float>(K.at<double>(1, 1));
        cx_ = static_cast<float>(K.at<double>(0, 2));
        cy_ = static_cast<float>(K.at<double>(1, 2));
    } else {
        // Estimated for 1280×720 USB webcam, ~90° horizontal FOV
        // fx = (width/2) / tan(45°) = width/2
        // Use --calib to pass actual calibrated K for metric accuracy
        fx_ = 640.0f;
        fy_ = 640.0f;
        cx_ = 640.0f;
        cy_ = 360.0f;
    }

    scale_x_ = static_cast<float>(BEV_W) / (2.0f * RANGE_LATERAL_M);
    scale_y_ = static_cast<float>(BEV_H - 40) / RANGE_FORWARD_M;

    ego_x_ = BEV_W / 2;
    ego_y_ = BEV_H - 35;
}

// ── Projection ────────────────────────────────────────────────────────────────

bool BEVVisualizer::project(float u, float v,
                             int& bev_x, int& bev_y,
                             float& dist_m) const
{
    float dv = v - cy_;
    if (dv <= 1.0f) return false;

    float Z = cam_h_ * fy_ / dv;
    float X = (u - cx_) * Z / fx_;

    if (Z < 0.5f || Z > RANGE_FORWARD_M) return false;
    if (std::fabs(X) > RANGE_LATERAL_M)  return false;

    dist_m = Z;

    bev_x = ego_x_ + static_cast<int>(X * scale_x_);
    bev_y = ego_y_ - static_cast<int>(Z * scale_y_);

    return bev_x >= 0 && bev_x < BEV_W &&
           bev_y >= 0 && bev_y < BEV_H;
}

// ── Background ────────────────────────────────────────────────────────────────

cv::Mat BEVVisualizer::makeBackground() const {
    cv::Mat canvas(BEV_H, BEV_W, CV_8UC3, BG_COLOR);

    // Road surface
    int road_px = static_cast<int>(3.75f * scale_x_);
    cv::rectangle(canvas,
                  cv::Point(ego_x_ - road_px, 0),
                  cv::Point(ego_x_ + road_px, ego_y_),
                  ROAD_COLOR, cv::FILLED);

    // Arc-based range rings — every 5m, label all of them
    for (int d = 5; d <= static_cast<int>(RANGE_FORWARD_M); d += 5) {
        int r_px = static_cast<int>(d * scale_y_);
        cv::ellipse(canvas,
                    cv::Point(ego_x_, ego_y_),
                    cv::Size(r_px, r_px),
                    0, 180, 360, GRID_COLOR, 1, cv::LINE_AA);

        int y = ego_y_ - r_px;
        if (y > 8) {
            std::string lbl = std::to_string(d) + "m";
            cv::putText(canvas, lbl,
                        cv::Point(ego_x_ + r_px + 3, y + 4),
                        cv::FONT_HERSHEY_SIMPLEX, 0.30,
                        GRID_TEXT, 1, cv::LINE_AA);
        }
    }

    // Lateral tick marks
    for (int x_m = -12; x_m <= 12; x_m += 4) {
        if (x_m == 0) continue;
        int bx = ego_x_ + static_cast<int>(x_m * scale_x_);
        if (bx < 5 || bx > BEV_W - 5) continue;
        cv::line(canvas,
                 cv::Point(bx, ego_y_),
                 cv::Point(bx, ego_y_ - 6),
                 GRID_COLOR, 1, cv::LINE_AA);
    }

    // FOV cone — angle derived from fx_ (matches actual projection)
    float half_fov = std::atan(cx_ / fx_);  // e.g. atan(640/640) = 45°
    int   fov_r    = static_cast<int>(RANGE_FORWARD_M * scale_y_);
    cv::Point ego_pt(ego_x_, ego_y_);
    cv::Point fov_l(ego_x_ - static_cast<int>(std::sin(half_fov) * fov_r),
                    ego_y_ - static_cast<int>(std::cos(half_fov) * fov_r));
    cv::Point fov_r_pt(ego_x_ + static_cast<int>(std::sin(half_fov) * fov_r),
                       ego_y_ - static_cast<int>(std::cos(half_fov) * fov_r));

    // Filled transparent FOV region
    std::vector<cv::Point> fov_pts = {ego_pt, fov_l, fov_r_pt};
    cv::Mat overlay = canvas.clone();
    cv::fillPoly(overlay, fov_pts, cv::Scalar(0x3f, 0xb9, 0x50));
    cv::addWeighted(overlay, 0.06, canvas, 0.94, 0, canvas);

    cv::line(canvas, ego_pt, fov_l,   FOV_COLOR, 1, cv::LINE_AA);
    cv::line(canvas, ego_pt, fov_r_pt, FOV_COLOR, 1, cv::LINE_AA);

    // Dashed center lane line
    int dash = static_cast<int>(3.0f * scale_y_);
    int gap  = static_cast<int>(2.5f * scale_y_);
    for (int y = ego_y_ - gap; y > 0; y -= dash + gap) {
        int y2 = std::max(0, y - dash);
        cv::line(canvas,
                 cv::Point(ego_x_, y),
                 cv::Point(ego_x_, y2),
                 LANE_COLOR, 1, cv::LINE_AA);
    }

    // ── Ego vehicle (proper car shape) ───────────────────────────────────────
    int ew = 18, el = 32;
    int ex = ego_x_, ey = ego_y_ - 4;

    // Body
    cv::rectangle(canvas,
                  cv::Point(ex - ew/2, ey - el),
                  cv::Point(ex + ew/2, ey),
                  EGO_FILL, cv::FILLED);

    // Roof (slightly narrower, offset forward)
    cv::rectangle(canvas,
                  cv::Point(ex - ew/2 + 3, ey - el + 7),
                  cv::Point(ex + ew/2 - 3, ey - 10),
                  EGO_ROOF, cv::FILLED);

    // Outline
    cv::rectangle(canvas,
                  cv::Point(ex - ew/2, ey - el),
                  cv::Point(ex + ew/2, ey),
                  cv::Scalar(180, 210, 255), 1);

    // Headlights
    cv::circle(canvas, cv::Point(ex - ew/2 + 3, ey - el + 3),
               2, cv::Scalar(255, 255, 200), cv::FILLED);
    cv::circle(canvas, cv::Point(ex + ew/2 - 3, ey - el + 3),
               2, cv::Scalar(255, 255, 200), cv::FILLED);

    // Bottom panel: border + title
    cv::rectangle(canvas,
                  cv::Point(0, BEV_H - 22),
                  cv::Point(BEV_W, BEV_H),
                  PANEL_COLOR, cv::FILLED);
    cv::line(canvas,
             cv::Point(0, BEV_H - 22),
             cv::Point(BEV_W, BEV_H - 22),
             GRID_COLOR, 1);
    cv::putText(canvas, "BEV  (ground plane projection, approx.)",
                cv::Point(6, BEV_H - 7),
                cv::FONT_HERSHEY_SIMPLEX, 0.28,
                TITLE_COLOR, 1, cv::LINE_AA);

    return canvas;
}

// ── Draw one detection ────────────────────────────────────────────────────────

void BEVVisualizer::drawDetection(cv::Mat& canvas,
                                  const Detection& det,
                                  float u, float v) const
{
    int   bev_x, bev_y;
    float dist_m;

    if (!project(u, v, bev_x, bev_y, dist_m)) return;

    cv::Scalar color =
        (YOLO26Decoder::CLASS_COLORS.size() > (size_t)det.class_id)
        ? YOLO26Decoder::CLASS_COLORS[det.class_id]
        : cv::Scalar(255, 255, 255);

    // Estimate object width in BEV from box pixel width
    float box_w_px = det.x2 - det.x1;
    float est_w_m  = std::max(0.6f,
                        std::min(3.5f, box_w_px * dist_m / fx_));
    int   hw       = std::max(5, static_cast<int>(est_w_m * scale_x_ * 0.5f));
    int   hl       = std::max(7, hw + 4);

    // Filled rounded rectangle (simulate object footprint)
    cv::rectangle(canvas,
                  cv::Point(bev_x - hw, bev_y - hl),
                  cv::Point(bev_x + hw, bev_y + hl),
                  color, cv::FILLED);
    cv::rectangle(canvas,
                  cv::Point(bev_x - hw, bev_y - hl),
                  cv::Point(bev_x + hw, bev_y + hl),
                  cv::Scalar(255, 255, 255), 1);

    // Subtle connector line from ego
    cv::line(canvas,
             cv::Point(ego_x_, ego_y_),
             cv::Point(bev_x, bev_y),
             cv::Scalar(color[0]*0.25, color[1]*0.25, color[2]*0.25),
             1, cv::LINE_AA);

    // Label
    std::ostringstream ss;
    ss << det.class_name.substr(0, 3) << " "
       << std::fixed << std::setprecision(0) << dist_m << "m";

    int baseline = 0;
    cv::Size ts = cv::getTextSize(
        ss.str(), cv::FONT_HERSHEY_SIMPLEX, 0.28, 1, &baseline);

    int lx = std::max(2, std::min(bev_x - ts.width/2, BEV_W - ts.width - 2));
    int ly = std::max(ts.height + 2, bev_y - hl - 3);

    // Label background
    cv::rectangle(canvas,
                  cv::Point(lx - 1, ly - ts.height - 1),
                  cv::Point(lx + ts.width + 1, ly + 1),
                  cv::Scalar(0x0d, 0x11, 0x17), cv::FILLED);

    cv::putText(canvas, ss.str(),
                cv::Point(lx, ly),
                cv::FONT_HERSHEY_SIMPLEX, 0.28,
                cv::Scalar(220, 220, 220), 1, cv::LINE_AA);
}

// ── Render ────────────────────────────────────────────────────────────────────

cv::Mat BEVVisualizer::render(const std::vector<Detection>& dets,
                               int img_w, int img_h)
{
    // Adjust intrinsics if resolution differs from 1280×720 default
    if (img_w != 1280) {
        cx_ = img_w  * 0.5f;
        cy_ = img_h  * 0.5f;
        fx_ = fy_ = cx_;   // 90° HFOV: fx = width/2
    }

    cv::Mat canvas = makeBackground();

    // Sort by distance (far first) so close objects render on top
    std::vector<std::pair<float,int>> order;
    for (int i = 0; i < (int)dets.size(); i++) {
        float u = (dets[i].x1 + dets[i].x2) * 0.5f;
        float v =  dets[i].y2;
        float dv = v - cy_;
        float dist = (dv > 1.0f) ? cam_h_ * fy_ / dv : 999.0f;
        order.push_back({dist, i});
    }
    std::sort(order.begin(), order.end(),
              [](auto& a, auto& b){ return a.first > b.first; });

    for (auto& [dist, i] : order) {
        float u = (dets[i].x1 + dets[i].x2) * 0.5f;
        float v =  dets[i].y2;
        drawDetection(canvas, dets[i], u, v);
    }

    return canvas;
}

// ── Side by side ──────────────────────────────────────────────────────────────

cv::Mat BEVVisualizer::sideBySide(const cv::Mat& cam,
                                   const cv::Mat& bev) const
{
    int target_h = cam.rows;
    float scale  = static_cast<float>(target_h) / bev.rows;
    int   bev_w  = static_cast<int>(bev.cols * scale);

    cv::Mat bev_scaled;
    cv::resize(bev, bev_scaled, cv::Size(bev_w, target_h),
               0, 0, cv::INTER_LINEAR);

    cv::Mat divider(target_h, 2, CV_8UC3, cv::Scalar(48, 54, 61));

    cv::Mat combined;
    cv::hconcat(std::vector<cv::Mat>{cam, divider, bev_scaled}, combined);
    return combined;
}
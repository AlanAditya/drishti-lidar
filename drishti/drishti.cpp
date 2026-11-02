#pragma once
// drishti: LiDAR perception on matrix, with SalsaNext (MLX) for the segmentation. Every step is a
// function or object with matrices in and out, and its settings in the open, so a program runs the
// steps itself: in its own order, reading or changing what passes between them, swapping one out,
// and choosing where evaluation happens.
//
//   points [N, 4]                  KittiSequence::points
//   xyz -> pixel [N]               spherical_projection     each point's pixel (H*W = none)
//   pixel, range -> owner [H*W]    closest_per_pixel        the point each pixel shows (-1 = empty)
//   points, range, owner -> image  range_image              [1, H, W, 5] in metres
//   image -> probs [1, H, W, C]    SalsaNext::inference
//   per-pixel values -> [N]        to_points                each point takes its pixel's value
//   pixel, owner -> owns [N]       owns_pixel               the point is the one its pixel shows
//   points, group, cls -> cells    FoveatedGrid::bin        [G, 5] the 2.5D grid
//   xyz, mask -> id [N]            cluster                  object instances (-1 = not kept)
//   xyz, id, prob -> boxes [B, 9]  fit_boxes
//   boxes -> boxes [B, 11]         Tracker::update          + track id, speed
//
// Evaluation: the graph functions (projection, owner, image, to_points, owns_pixel, bin) only
// build graph. Nothing runs until matrix::eval, or a step that reads on the host: SalsaNext::
// inference (its image goes into MLX), cluster, fit_boxes and Tracker::update.
//
// Plain C++ on the matrix API (no Objective-C, no Metal calls). Include it from the one file that
// uses it; it isn't compiled on its own. FoveatedGrid's kernels are in AI/FoveatedGrid.metal.
// macOS only (MLX).

#include <TargetConditionals.h>
#if !TARGET_OS_IPHONE

#include "../matrix/matrix.h"
#include "BoundingBoxes.h"
#include "FoveatedGrid.h"
#include "Kitti.h"
#include "SalsaNext.h"
#include "Tracking.h"
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace drishti {

namespace mx = mlx::core;

namespace detail {
// `m` evaluated, of type `t`, row-major contiguous, for host loops: `m` itself when it already is,
// otherwise a copy made on the CPU (no GPU round trip for a strided view such as points[:, 0:3]).
inline matrix host(const matrix& m, dtype t) {
    matrix h = m;
    matrix::eval(h);
    if (h.type == t && !(h.flags & NON_CONTIGUOUS_FLAG)) return h;
    matrix c = h.astype(t, true);
    c.eval_cpu();
    return c;
}
} // namespace detail

// ---- the sensor -------------------------------------------------------------------------------

// A spinning LiDAR's range image: `height` rows (one per laser) spread over [fov_down, fov_up]
// degrees of pitch, `width` columns over the full turn.
struct Sensor {
    int height = 64, width = 2048;            // Velodyne HDL-64
    float fov_up = 3.0f, fov_down = -25.0f;   // degrees
    size_m pixels() const { return (size_m)height * width; }
};

// ---- KITTI ------------------------------------------------------------------------------------

// A SemanticKITTI sequence (kitti::Sequence: scan count, poses, paths) whose scans come out as matrices.
class KittiSequence : public kitti::Sequence {
public:
    using kitti::Sequence::Sequence;

    // Scan `i` (its .bin file) straight into a contiguous [N, 4] Float: x, y, z, remission.
    matrix points(size_t i) const {
        const std::string path = this->path("velodyne", i, ".bin");
        FILE* fp = std::fopen(path.c_str(), "rb");
        if (!fp) throw std::runtime_error("drishti: can't open " + path);
        std::fseek(fp, 0, SEEK_END);
        const size_m n = (size_m)(std::ftell(fp) / 16);
        std::fseek(fp, 0, SEEK_SET);
        if (n == 0) { std::fclose(fp); throw std::runtime_error("drishti: empty scan " + path); }
        matrix pts = matrix::withShape({n, 4}, dtype::Float);
        pts.begin_refcount();
        const size_t read = std::fread(pts.buffer, 16, n, fp);
        std::fclose(fp);
        if (read != n) throw std::runtime_error("drishti: truncated scan " + path);
        return pts;
    }

    // Scan `i`'s ground truth (its .label file) as SalsaNext's 20 classes (kitti::to_train_class): [N] Int32.
    matrix classes(size_t i) const {
        const std::string path = this->path("labels", i, ".label");
        FILE* fp = std::fopen(path.c_str(), "rb");
        if (!fp) throw std::runtime_error("drishti: can't open " + path);
        std::fseek(fp, 0, SEEK_END);
        const size_m n = (size_m)(std::ftell(fp) / 4);
        std::fseek(fp, 0, SEEK_SET);
        if (n == 0) { std::fclose(fp); throw std::runtime_error("drishti: empty labels " + path); }
        matrix cls = matrix::withShape({n}, dtype::Int32);
        cls.begin_refcount();
        const size_t read = std::fread(cls.buffer, 4, n, fp);
        std::fclose(fp);
        if (read != n) throw std::runtime_error("drishti: truncated labels " + path);
        int32_t* c = (int32_t*)cls.buffer;
        for (size_m k = 0; k < n; ++k) c[k] = kitti::to_train_class((uint16_t)((uint32_t)c[k] & 0xFFFF));
        return cls;
    }
};

// SalsaNext's classes -> the planner's groups (PS 26053, kitti::group_of): 0 unknown, 1 terrain,
// 2 static, 3 dynamic. [classes] Int32; group = table.take(cls, 0).
inline matrix kitti_group_table(int classes = 20) {
    matrix t = matrix::withShape({(size_m)classes}, dtype::Int32);
    t.begin_refcount();
    for (int c = 0; c < classes; ++c) ((int32_t*)t.buffer)[c] = (int32_t)kitti::group_of(c);
    return t;
}

// ---- projection -------------------------------------------------------------------------------

// Each point's pixel in the sensor's range image, as SalsaNext / RangeNet++ project (ai::project):
// column from yaw, row from pitch within [fov_down, fov_up], floor and clamp. xyz [N, 3] Float
// (a view of points is fine) -> [N] Int32 row * W + col; H * W for a point at the origin (no
// direction). Lazy.
inline matrix spherical_projection(const matrix& xyz_in, const Sensor& s) {
    matrix xyz = xyz_in;
    const int H = s.height, W = s.width;
    matrix x = xyz[R(), R(0, 1)], y = xyz[R(), R(1, 2)], z = xyz[R(), R(2, 3)];
    const float up = s.fov_up / 180.0f * (float)M_PI, down = s.fov_down / 180.0f * (float)M_PI;
    const float fov = std::fabs(down) + std::fabs(up);
    matrix d = matrix::sqrt((xyz * xyz).sum(-1, true));                                      // [N, 1]
    matrix yaw = matrix::arctan2(y, x) * -1.0f;
    matrix pitch = matrix::arcsin(z / d);
    matrix col = matrix::floor((yaw / (float)M_PI + 1.0f) * 0.5f * (float)W).clamp(0, W - 1).astype(dtype::Int32);
    matrix row = matrix::floor((1.0f - (pitch + std::fabs(down)) / fov) * (float)H).clamp(0, H - 1).astype(dtype::Int32);
    return matrix::where(d > 0.0f, row * W + col, (int)s.pixels()).flatten();
}

// The point each pixel shows: the closest one (the lowest index on a tie), as SalsaNext's input
// keeps it. pixel [N] Int32, range [N] Float -> [H * W] Int32, -1 where no point landed. Lazy.
// Two scatter MINs: the smallest range per pixel, then the smallest index among the points at it.
inline matrix closest_per_pixel(const matrix& pixel, const matrix& range, const Sensor& s) {
    const size_m N = const_cast<matrix&>(pixel).shape()[0], P = s.pixels();
    matrix d = const_cast<matrix&>(range).reshape((int)N);
    matrix best = matrix::scatter(matrix::full({P + 1}, INFINITY, dtype::Float), pixel, d, 0, ScatterMode::MIN);
    matrix ids = matrix::arange(0, N, 1, dtype::Int32);
    matrix winner = matrix::where(matrix::equal(d, const_cast<matrix&>(best).take(pixel, 0)), pixel, (int)P);
    matrix owner_all = matrix::scatter(matrix::full({P + 1}, INT32_MAX, dtype::Int32), winner, ids, 0, ScatterMode::MIN);
    matrix owner = owner_all[R(0, P)];
    return matrix::where(owner < INT32_MAX, owner, -1);
}

// The range image: each pixel's point's range, x, y, z, remission, in metres (not normalised; the
// network does that), 0 in every channel where no point landed. points [N, 4], range [N],
// owner [H * W] -> [1, H, W, 5] Float. Lazy.
inline matrix range_image(const matrix& points, const matrix& range, const matrix& owner, const Sensor& s) {
    const size_m N = const_cast<matrix&>(points).shape()[0], P = s.pixels();
    matrix features = matrix::concat({const_cast<matrix&>(range).reshape((int)N, 1), points}, -1);   // [N, 5]
    matrix has = owner > -1;
    return matrix::where(has.reshape((int)P, 1), features.take(owner.clamp(0, N - 1), 0), 0.0f).reshape(1, s.height, s.width, 5);
}

// Each point takes its pixel's value: per-pixel values (anything with H * W elements, e.g.
// probs.argmax(-1)) -> [N]; 0 for a point with no pixel. Lazy.
inline matrix to_points(const matrix& values, const matrix& pixel) {
    matrix v = values.flatten();
    return matrix::concat({v, matrix::full({1}, 0, v.type)}, 0).take(pixel, 0);
}

// Whether each point is the one its pixel shows. The others only inherit a closer point's class
// (the wall behind a car becomes "car"). pixel [N], owner [H * W] -> [N] Bool. Lazy.
inline matrix owns_pixel(const matrix& pixel, const matrix& owner) {
    const size_m N = const_cast<matrix&>(pixel).shape()[0];
    matrix pad = matrix::concat({owner, matrix::full({1}, -1, dtype::Int32)}, 0);
    return matrix::equal(pad.take(pixel, 0), matrix::arange(0, N, 1, dtype::Int32));
}

// ---- segmentation -----------------------------------------------------------------------------

// SalsaNext (ai::SalsaNext, pretrained on SemanticKITTI): a range image in, per-pixel class
// probabilities out. The weights carry the sensor and the normalisation it was trained with.
class SalsaNext {
public:
    std::shared_ptr<ai::SalsaNext> model;
    matrix mean, stdv;   // [5]: range, x, y, z, remission

    explicit SalsaNext(const std::string& safetensors_path) : SalsaNext(std::make_shared<ai::SalsaNext>(safetensors_path)) {}
    explicit SalsaNext(std::shared_ptr<ai::SalsaNext> m) : model(std::move(m)) {
        mean = matrix::of<float>({model->means[0], model->means[1], model->means[2], model->means[3], model->means[4]});
        stdv = matrix::of<float>({model->stds[0], model->stds[1], model->stds[2], model->stds[3], model->stds[4]});
    }

    Sensor sensor() const { return {model->height, model->width, model->fov_up, model->fov_down}; }
    int classes() const { return model->classes; }

    // image [1, H, W, 5] (range_image) -> probs [1, H, W, C] Float. Normalises each channel with
    // mean / stdv, keeping empty pixels (range 0) at 0; evaluates; runs the network (MLX); copies
    // the probabilities back into a plain matrix.
    matrix inference(const matrix& image) const {
        const int H = model->height, W = model->width;
        const size_m P = (size_m)H * W, C = (size_m)model->classes;
        matrix flat = const_cast<matrix&>(image).reshape((int)P, 5);
        matrix valid = flat[R(), R(0, 1)] > 0.0f;                                           // [P, 1]
        matrix norm = matrix::where(valid, (flat - mean) / stdv, 0.0f);
        matrix::eval(norm);
        mx::array in((const float*)norm.buffer, {1, H, W, 5}, mx::float32);
        mx::array p = (*model)(in);
        mx::eval(p);
        matrix probs = matrix::withShape({1, (size_m)H, (size_m)W, C}, dtype::Float);
        probs.begin_refcount();
        std::memcpy(probs.buffer, p.data<float>(), (size_t)P * C * sizeof(float));
        return probs;
    }
};

// ---- the foveated 2.5D grid -------------------------------------------------------------------

// The variable-resolution elevation map of AI/FoveatedGrid.h (square rings of cells, fine near the
// sensor, coarse far away), built with matrix: two gpu_func kernels around four scatters. A cell is
// [zmin, zmax, count, group, class]: an empty one +inf, -inf, 0, 0, 0; `group` and `class` are
// those of the highest-group point in it, the earliest on a tie.
class FoveatedGrid {
public:
    foveated::Grid layout;   // the rings; cell i covers layout.cell_bounds(i)

    explicit FoveatedGrid(const std::vector<foveated::RingSpec>& rings = foveated::default_rings()) : layout(rings) {
        const auto& r = layout.rings();
        rings_f = matrix::withShape({(size_m)r.size(), 2}, dtype::Float);
        rings_i = matrix::withShape({(size_m)r.size(), 3}, dtype::Int32);
        rings_f.begin_refcount();
        rings_i.begin_refcount();
        for (size_t k = 0; k < r.size(); ++k) {
            ((float*)rings_f.buffer)[k * 2] = r[k].cell;
            ((float*)rings_f.buffer)[k * 2 + 1] = r[k].half;
            ((int32_t*)rings_i.buffer)[k * 3] = r[k].n;
            ((int32_t*)rings_i.buffer)[k * 3 + 1] = r[k].band;
            ((int32_t*)rings_i.buffer)[k * 3 + 2] = (int32_t)r[k].offset;
        }
    }

    size_m cells() const { return (size_m)layout.cell_count(); }

    // points [N, 4] Float (contiguous), group and cls [N] Int32 -> cells [G, 5] Float. Lazy.
    // Per point: its cell and its values encoded for uint32 atomic MAX; four scatters reduce them
    // per cell; per cell they're decoded.
    matrix bin(const matrix& points, const matrix& group, const matrix& cls) const {
        const size_m N = const_cast<matrix&>(points).shape()[0], G = cells();
        matrix enc = matrix::host_op({4, N}, dtype::UInt32, gpu_func("foveated_grid_points", {N}), {points, group, rings_f, rings_i});
        auto row_of = [&](int r) { return enc[R(r, r + 1)].flatten(); };
        auto target = [&] { return matrix::full({G + 1}, 0, dtype::UInt32); };              // slot G: outside the grid
        matrix k = row_of(0);
        matrix zmax_key = matrix::scatter(target(), k, row_of(1), 0, ScatterMode::MAX);
        matrix zmin_key = matrix::scatter(target(), k, row_of(2), 0, ScatterMode::MAX);
        matrix counts = matrix::scatter(target(), k, matrix::full({N}, 1, dtype::UInt32), 0, ScatterMode::ADD);
        matrix gkey = matrix::scatter(target(), k, row_of(3), 0, ScatterMode::MAX);
        return matrix::host_op({G, 5}, dtype::Float, gpu_func("foveated_grid_cells", {G}), {zmax_key, zmin_key, counts, gkey, cls});
    }

private:
    matrix rings_f, rings_i;   // the rings for foveated_grid_points: [R, 2] Float, [R, 3] Int32
};

// ---- objects ----------------------------------------------------------------------------------

// The columns of a boxes matrix, one box per row (Float). fit_boxes gives the first 9,
// Tracker::update adds the last 2.
enum BoxCol { BoxCX, BoxCY, BoxCZ, BoxLength, BoxWidth, BoxHeight, BoxYaw, BoxPoints, BoxScore,
              BoxTrack,      // the track's id (track::Track::id), stable across scans
              BoxSpeed,      // m/s in the world frame; NaN until its track has a velocity
              BoxCols };
constexpr int kFitCols = BoxTrack;   // 9

// Object instances in bird's-eye view: the points `mask` keeps are binned into `cell` x `cell`
// metre squares, and touching squares (diagonals too) are one object, so points closer than about
// one cell join and gaps wider than about two split. xyz [N, 3], mask [N] Bool -> [N] Int32 object
// id (0, 1, ... in order of each object's first point), -1 where not kept. Host code, O(N);
// evaluates its inputs.
inline matrix cluster(const matrix& xyz, const matrix& mask, float cell) {
    matrix p = detail::host(xyz, dtype::Float);
    matrix m = detail::host(mask, dtype::Bool);
    const size_m n = p.shape()[0];
    const float* q = (const float*)p.buffer;
    const bool* keep = (const bool*)m.buffer;
    matrix id = matrix::withShape({n}, dtype::Int32);
    id.begin_refcount();
    int32_t* out = (int32_t*)id.buffer;

    // Occupied squares, in order of their first point, and the points in each.
    std::unordered_map<int64_t, int> square_of;
    std::vector<std::pair<int, int>> square_xy;
    std::vector<int32_t> square;                   // per point: its square, -1 if not kept
    square.assign(n, -1);
    for (size_m i = 0; i < n; ++i) {
        if (!keep[i]) continue;
        const int ix = (int)std::floor(q[i * 3] / cell), iy = (int)std::floor(q[i * 3 + 1] / cell);
        auto [it, added] = square_of.try_emplace(bbox::detail::cell_key(ix, iy), (int)square_xy.size());
        if (added) square_xy.push_back({ix, iy});
        square[i] = it->second;
    }
    // Flood-fill touching squares into objects, numbered in that order.
    std::vector<int32_t> object(square_xy.size(), -1);
    std::vector<int> stack;
    int32_t next = 0;
    for (size_t start = 0; start < square_xy.size(); ++start) {
        if (object[start] >= 0) continue;
        object[start] = next;
        stack.assign(1, (int)start);
        while (!stack.empty()) {
            const auto [ix, iy] = square_xy[stack.back()];
            stack.pop_back();
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx) {
                    auto it = square_of.find(bbox::detail::cell_key(ix + dx, iy + dy));
                    if (it != square_of.end() && object[it->second] < 0) {
                        object[it->second] = next;
                        stack.push_back(it->second);
                    }
                }
        }
        ++next;
    }
    for (size_m i = 0; i < n; ++i) out[i] = square[i] < 0 ? -1 : object[square[i]];
    return id;
}

// fit_boxes' filters and L-shape search (bbox::Params without the clustering cell).
struct BoxParams {
    int min_points = 15;          // smaller objects are dropped (stray mislabelled points)
    float min_score = 0.9f;       // mean probability needed
    float min_width = 0.3f;       // metres; thinner objects are slivers
    float angle_step_deg = 1.0f;  // L-shape heading search step
    float min_closeness = 0.01f;  // caps one point's L-shape score (d0 in Zhang et al.)
};

// One oriented box per object of `id` (cluster), as bbox::fit_boxes fits them: the L-shape
// heading whose rectangle the points hug most closely (a LiDAR sees only the sides facing it),
// height from the lowest to the highest point. Objects with fewer than min_points points, a mean
// `prob` below min_score, or thinner than min_width are dropped. xyz [N, 3], id [N] Int32, prob
// [N] Float -> [B, 9] Float (BoxCX ... BoxScore), in object order. Host code; evaluates its inputs.
inline matrix fit_boxes(const matrix& xyz, const matrix& id, const matrix& prob, const BoxParams& params = {}) {
    matrix p = detail::host(xyz, dtype::Float);
    matrix o = detail::host(id, dtype::Int32);
    matrix c = detail::host(prob, dtype::Float);
    const size_m n = p.shape()[0];
    const float* q = (const float*)p.buffer;
    const int32_t* obj = (const int32_t*)o.buffer;
    const float* conf = (const float*)c.buffer;

    // Each object's points, in point order (a counting sort by id).
    int32_t objects = 0;
    for (size_m i = 0; i < n; ++i) objects = std::max(objects, obj[i] + 1);
    std::vector<uint32_t> start(objects + 1, 0), members;
    for (size_m i = 0; i < n; ++i) if (obj[i] >= 0) start[obj[i] + 1]++;
    for (int32_t k = 0; k < objects; ++k) start[k + 1] += start[k];
    members.resize(start[objects]);
    std::vector<uint32_t> fill(start.begin(), start.end() - 1);
    for (size_m i = 0; i < n; ++i) if (obj[i] >= 0) members[fill[obj[i]]++] = (uint32_t)i;

    bbox::Params lp;
    lp.angle_step_deg = params.angle_step_deg;
    lp.min_closeness = params.min_closeness;
    std::vector<float> rows, pts;
    for (int32_t k = 0; k < objects; ++k) {
        const uint32_t* m = members.data() + start[k];
        const size_t count = start[k + 1] - start[k];
        if ((int)count < params.min_points) continue;
        double sum = 0;
        for (size_t j = 0; j < count; ++j) sum += conf[m[j]];
        const float score = (float)(sum / count);
        if (score < params.min_score) continue;

        pts.resize(count * 2);
        float zmin = INFINITY, zmax = -INFINITY;
        for (size_t j = 0; j < count; ++j) {
            const float* v = &q[m[j] * 3];
            pts[j * 2] = v[0];
            pts[j * 2 + 1] = v[1];
            zmin = std::min(zmin, v[2]);
            zmax = std::max(zmax, v[2]);
        }
        float theta = bbox::detail::l_shape_heading(pts, lp);
        const float cs = std::cos(theta), sn = std::sin(theta);
        float min1 = INFINITY, max1 = -INFINITY, min2 = INFINITY, max2 = -INFINITY;
        for (size_t j = 0; j < count; ++j) {
            const float a = pts[j * 2] * cs + pts[j * 2 + 1] * sn, b = -pts[j * 2] * sn + pts[j * 2 + 1] * cs;
            min1 = std::min(min1, a); max1 = std::max(max1, a);
            min2 = std::min(min2, b); max2 = std::max(max2, b);
        }
        const float m1 = 0.5f * (min1 + max1), m2 = 0.5f * (min2 + max2);
        float length = max1 - min1, width = max2 - min2;
        if (width > length) {                       // keep `length` the long side
            std::swap(length, width);
            theta += (float)M_PI_2;
        }
        if (width < params.min_width) continue;
        const float row[kFitCols] = {m1 * cs - m2 * sn, m1 * sn + m2 * cs, 0.5f * (zmin + zmax),
                                     length, width, zmax - zmin, theta, (float)count, score};
        rows.insert(rows.end(), row, row + kFitCols);
    }
    const size_m b = (size_m)(rows.size() / kFitCols);
    matrix boxes = matrix::withShape({b, (size_m)kFitCols}, dtype::Float);
    if (b == 0) return boxes;
    boxes.begin_refcount();
    std::memcpy(boxes.buffer, rows.data(), rows.size() * sizeof(float));
    return boxes;
}

// Boxes followed across scans in the world frame (track::Tracker: nearest-neighbour matching with
// a constant-velocity prediction), for each object's speed.
class Tracker {
public:
    track::Tracker tracker;

    explicit Tracker(const track::Params& p = {}) : tracker(p) {}

    // boxes [B, 9+] (fit_boxes), `pose` mapping this scan into the world (kitti::Sequence::pose),
    // `t` its time in seconds -> [B, 11]: the boxes plus BoxTrack and BoxSpeed. Feed consecutive
    // scans only; reset() after a jump.
    matrix update(const matrix& boxes, simd_float4x4 pose, double t) {
        matrix in = detail::host(boxes, dtype::Float);
        const size_m b = in.dims ? in.shape()[0] : 0, cols = b ? in.shape()[1] : 0;
        std::vector<float> world;
        for (size_m j = 0; j < b; ++j) {
            const float* r = (const float*)in.buffer + j * cols;
            const simd_float4 w = simd_mul(pose, simd_make_float4(r[BoxCX], r[BoxCY], r[BoxCZ], 1));
            world.push_back(w.x);
            world.push_back(w.y);
        }
        const std::vector<int> ids = tracker.update(world, t);
        matrix out = matrix::withShape({b, (size_m)BoxCols}, dtype::Float);
        if (b == 0) return out;
        out.begin_refcount();
        for (size_m j = 0; j < b; ++j) {
            const float* r = (const float*)in.buffer + j * cols;
            float* o = (float*)out.buffer + j * BoxCols;
            std::memcpy(o, r, kFitCols * sizeof(float));
            const track::Track& tr = tracker.tracks()[ids[j]];
            o[BoxTrack] = (float)tr.id;
            o[BoxSpeed] = tr.has_velocity ? tr.speed() : NAN;
        }
        return out;
    }

    void reset() { tracker.reset(); }
};

} // namespace drishti

#endif // !TARGET_OS_IPHONE

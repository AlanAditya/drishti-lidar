#pragma once
// drishti's outputs as what the WorldOf3D renderer's controllers take: tiles for a
// TriangleMeshController, positions and colours for a PointCloudController, box transforms and
// label texts for a LabelledBoxController. Matrices in, matrices out (the labels' strings aside);
// no renderer types.
//
// KITTI is x forward, y left, z up; the renderer's viewer is y up, looking down -z: (x, y, z) -> (-y, z, -x).
#include "drishti.cpp"
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace drishti::draw {

// Group colours: unknown, terrain, static, dynamic.
inline constexpr float kGroupRGB[4][3] = {{0.35f, 0.35f, 0.35f}, {0.30f, 0.75f, 0.35f}, {0.40f, 0.55f, 0.95f}, {1.0f, 0.30f, 0.20f}};

inline simd_float3 to_view(float x, float y, float z) { return simd_make_float3(-y, z, -x); }

// A palette: one RGB row per label, [K, 3] Float in [0, 1]. Tiles and points look their label up in it.
inline matrix group_palette() {
    matrix p = matrix::withShape({4, 3}, dtype::Float);
    p.begin_refcount();
    std::copy_n(&kGroupRGB[0][0], 12, (float*)p.buffer);
    return p;
}

// SemanticKITTI's colour for each of the network's classes (from the weights' metadata).
inline matrix class_palette(const SalsaNext& net) {
    const size_m C = (size_m)net.classes();
    matrix p = matrix::withShape({C, 3}, dtype::Float);
    p.begin_refcount();
    std::copy_n(net.model->class_colors.data(), C * 3, (float*)p.buffer);
    return p;
}

// Which label of a cell (FoveatedGrid::bin's columns) the tiles are coloured by.
enum class TileColour { Group = 3, Class = 4 };

struct Mesh {
    matrix vertices;   // [V, 3] Float
    matrix colours;    // [V, 4] Float RGBA
    matrix indices;    // [3T] UInt32
    size_m count = 0;  // tiles = occupied cells (V = 4 * count)
};

// One flat tile per occupied cell of `cells` ([G, 5], FoveatedGrid::bin) at the cell's highest point,
// a little smaller than the cell so the 5 cm / 40 cm structure shows. Coloured by the cell's group or
// class, `by`, looked up in `palette`, and brighter for taller cells (the sensor is 1.73 m above the road).
inline Mesh tiles(const FoveatedGrid& grid, const matrix& cells, const matrix& palette, TileColour by = TileColour::Class) {
    Mesh m;
    const size_m G = grid.cells();
    const float* cl = (const float*)cells.buffer;
    const float* pal = (const float*)palette.buffer;
    const int column = (int)by;
    for (size_m i = 0; i < G; ++i) m.count += cl[i * 5 + 2] != 0;
    if (m.count == 0) return m;
    m.vertices = matrix::withShape({m.count * 4, 3}, dtype::Float);
    m.colours = matrix::withShape({m.count * 4, 4}, dtype::Float);
    m.indices = matrix::withShape({m.count * 6}, dtype::UInt32);
    m.vertices.begin_refcount();
    m.colours.begin_refcount();
    m.indices.begin_refcount();
    float* V = (float*)m.vertices.buffer;
    float* C = (float*)m.colours.buffer;
    uint32_t* I = (uint32_t*)m.indices.buffer;
    uint32_t t = 0;
    for (size_m i = 0; i < G; ++i) {
        const float* c = cl + i * 5;                       // zmin, zmax, count, group, class
        if (c[2] == 0) continue;
        float x0, y0, s;
        grid.layout.cell_bounds(i, x0, y0, s);
        const float inset = s * 0.05f;
        const float xs[4] = {x0 + inset, x0 + s - inset, x0 + s - inset, x0 + inset};
        const float ys[4] = {y0 + inset, y0 + inset, y0 + s - inset, y0 + s - inset};
        const float* rgb = pal + (int)c[column] * 3;
        const float shade = 0.55f + 0.45f * std::clamp((c[1] + 1.73f) / 3.0f, 0.0f, 1.0f);
        for (int v = 0; v < 4; ++v) {
            const simd_float3 p = to_view(xs[v], ys[v], c[1]);
            float* o = V + (t * 4 + v) * 3;
            o[0] = p.x; o[1] = p.y; o[2] = p.z;
            float* q = C + (t * 4 + v) * 4;
            q[0] = rgb[0] * shade; q[1] = rgb[1] * shade; q[2] = rgb[2] * shade; q[3] = 1.0f;
        }
        const uint32_t base = t * 4, tri[6] = {base, base + 1, base + 2, base, base + 2, base + 3};
        std::copy(tri, tri + 6, I + t * 6);
        ++t;
    }
    return m;
}

struct Cloud {
    matrix positions;  // [N, 3] Float, the viewer's axes
    matrix colours;    // [N, 4] Float RGBA
};

// The points in the viewer's axes, each coloured by its label ([N] Int32: class or group) looked up
// in `palette`. Built as ops, evaluated, and detached: plain matrices for the renderer.
inline Cloud points(const matrix& points, const matrix& labels, const matrix& palette) {
    matrix p = points;
    matrix x = p[R(), R(0, 1)], y = p[R(), R(1, 2)], z = p[R(), R(2, 3)];
    const size_m K = const_cast<matrix&>(palette).shape()[0];
    matrix rgba = matrix::concat({palette, matrix::full({K, 1}, 1.0f, dtype::Float)}, -1);   // [K, 4]
    Cloud c;
    c.positions = matrix::concat({y * -1.0f, z * 1.0f, x * -1.0f}, -1);
    c.colours = rgba.take(labels, 0);
    matrix::eval(c.positions, c.colours);
    matrix::detach(c.positions, c.colours);
    return c;
}

// Unit cube -> each box of `b` ([B, BoxCols], Tracker::update), in the viewer: [B, 4, 4] Float, the
// column-vector transforms WireBoxController::update takes. Columns 0-2 are the box's length,
// height and width axes, column 3 its centre.
inline matrix boxes(const matrix& b) {
    const size_m n = const_cast<matrix&>(b).shape()[0];
    matrix out = matrix::withShape({n, 4, 4}, dtype::Float);
    if (n == 0) return out;
    out.begin_refcount();
    for (size_m j = 0; j < n; ++j) {
        const float* r = (const float*)b.buffer + j * BoxCols;
        const float c = std::cos(r[BoxYaw]), s = std::sin(r[BoxYaw]);
        simd_float4x4 m;
        m.columns[0] = simd_make_float4(to_view(c * r[BoxLength], s * r[BoxLength], 0), 0);
        m.columns[1] = simd_make_float4(to_view(0, 0, r[BoxHeight]), 0);
        m.columns[2] = simd_make_float4(to_view(-s * r[BoxWidth], c * r[BoxWidth], 0), 0);
        m.columns[3] = simd_make_float4(to_view(r[BoxCX], r[BoxCY], r[BoxCZ]), 1);
        out.SIMD_MAT((int)j) = m;
    }
    return out;
}

// One label per box (LabelledBoxController::update's texts): "CAR", then "CAR 23 KM/H" once its
// track has a speed, or `still` ("STOPPED") under `still_speed` m/s.
inline std::vector<std::string> speed_texts(matrix& b, const std::string& name, float still_speed, const std::string& still) {
    std::vector<std::string> texts;
    const size_m n = b.shape()[0];
    for (size_m j = 0; j < n; ++j) {
        const float v = ((const float*)b.buffer)[j * BoxCols + BoxSpeed];
        texts.push_back(std::isnan(v) ? name : name + " " + (v < still_speed ? still : std::to_string((int)std::lround(v * 3.6f)) + " KM/H"));
    }
    return texts;
}

} // namespace drishti::draw

// The foveated 2.5D grid (AI/FoveatedGrid.h) as two gpu_func kernels for drishti::FoveatedGrid::bin
// (AI/drishti.cpp) and kitti::Pipeline::grid (AI/KittiPipeline.cpp), so the grid is a handful of
// nodes instead of ~100 ops until matrix fuses ops itself. Between them, four scatter MAX / ADD ops
// into zero-filled targets do the per-cell reductions:
//   foveated_grid_points (per point): the cell the point falls in (FoveatedGrid::cell_index, ring by
//     ring) and its values encoded so one uint32 atomic MAX does each reduction. out is [4, N]:
//       row 0  cell index (G = outside the grid)
//       row 1  z as an order-preserving uint         -> MAX = zmax
//       row 2  its bitwise complement                -> MAX = zmin
//       row 3  group << 24 | (0xFFFFFF - i)          -> MAX = highest group, earliest point (i < 2^24)
//     Every encoding of a real z is > 0, so a zero-filled target means "no point yet".
//   foveated_grid_cells (per cell): decodes the four reductions to [zmin, zmax, count, group, class]
//     as foveated::Cell holds them (an empty cell: +inf, -inf, 0, 0, 0).
#include <metal_stdlib>
#include "../ComputeShaders/Utils.h"

using namespace metal;

// Order-preserving float <-> uint: a < b as floats <=> ordered(a) < ordered(b) as uints.
static inline uint ordered(float f) { uint u = as_type<uint>(f); return (u & 0x80000000u) ? ~u : (u | 0x80000000u); }
static inline float unordered(uint o) { return as_type<float>((o & 0x80000000u) ? (o & 0x7FFFFFFFu) : ~o); }

// rings_f: [R, 2] (cell size, half extent); rings_i: [R, 3] (cells per side n, band b, offset).
kernel void foveated_grid_points(TENSOR_ARG(out, uint, 0), TENSOR_ARG(pts, const float, 1), TENSOR_ARG(groups, const int, 2),
                              TENSOR_ARG(rings_f, const float, 3), TENSOR_ARG(rings_i, const int, 4),
                              uint i [[thread_position_in_grid]]) {
    const uint N = pts_shape[0];
    const int R = (int)rings_f_shape[0];
    const int nl = rings_i[(R - 1) * 3], hl = nl - 2 * rings_i[(R - 1) * 3 + 1];
    const uint G = (uint)rings_i[(R - 1) * 3 + 2] + (uint)(nl * nl - hl * hl);

    const float x = pts[i * 4], y = pts[i * 4 + 1], z = pts[i * 4 + 2];
    const float ax = fabs(x), ay = fabs(y), reach = max(ax, ay);
    uint k = G;
    for (int r = 0; r < R; ++r) {
        const float cell = rings_f[r * 2], edge = rings_f[r * 2 + 1];
        if (reach >= edge) continue;
        const int n = rings_i[r * 3], b = rings_i[r * 3 + 1], offset = rings_i[r * 3 + 2];
        int col = clamp((int)floor(precise::divide(x + edge, cell)), 0, n - 1);
        int row = clamp((int)floor(precise::divide(y + edge, cell)), 0, n - 1);
        if (row >= b && row < n - b && col >= b && col < n - b) {          // exactly on the hole's edge
            if (ay >= ax) row = y < 0 ? b - 1 : n - b;
            else col = x < 0 ? b - 1 : n - b;
        }
        int local;
        if (row < b) local = row * n + col;
        else if (row >= n - b) local = b * n + (row - (n - b)) * n + col;
        else local = 2 * b * n + (row - b) * 2 * b + (col < b ? col : col - (n - b) + b);
        k = (uint)(offset + local);
        break;
    }
    out[i] = k;
    out[N + i] = ordered(z);
    out[2 * N + i] = ~ordered(z);
    out[3 * N + i] = ((uint)groups[i] << 24) | (0xFFFFFFu - i);
}

// out: [G, 5] = zmin, zmax, count (capped at 65535), group, class of the winning point.
kernel void foveated_grid_cells(TENSOR_ARG(out, float, 0), TENSOR_ARG(zmax_key, const uint, 1), TENSOR_ARG(zmin_key, const uint, 2),
                             TENSOR_ARG(count, const uint, 3), TENSOR_ARG(gkey, const uint, 4), TENSOR_ARG(cls, const int, 5),
                             uint c [[thread_position_in_grid]]) {
    device float* o = out + c * 5;
    const uint n = count[c];
    if (n == 0) {
        o[0] = as_type<float>(0x7F800000u); o[1] = as_type<float>(0xFF800000u); o[2] = 0; o[3] = 0; o[4] = 0;
        return;
    }
    const uint g = gkey[c];
    o[0] = unordered(~zmin_key[c]);
    o[1] = unordered(zmax_key[c]);
    o[2] = (float)min(n, 65535u);
    o[3] = (float)(g >> 24);
    o[4] = (float)cls[0xFFFFFFu - (g & 0xFFFFFFu)];
}

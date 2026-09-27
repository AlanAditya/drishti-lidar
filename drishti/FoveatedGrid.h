#pragma once
// Variable-resolution 2.5D grid (PS 26053): an elevation map with a semantic layer, fine near the
// sensor and coarse far away.
//
// Layout: nested square rings. Ring 0 is a full square of the finest cells around the sensor;
// every ring after it is the square band between the previous ring's edge and its own, with a
// larger cell size. The default is the brief's 5 cm near / ~50 cm far:
//
//     ring 0:  5 cm cells, |x|,|y| < 10 m     ring 2: 20 cm cells, 20 m .. 40 m
//     ring 1: 10 cm cells, 10 m .. 20 m       ring 3: 40 cm cells, 40 m .. 100 m
//
// No alignment errors, no lost points: every ring edge is a whole number of cells of both rings
// that meet there (checked in the constructor), so the rings tile the area exactly - no gaps, no
// overlaps - and each point inside the outer edge goes into exactly one cell. Rings are squares
// (distance = max(|x|, |y|)) so cells stay axis-aligned for a Cartesian planner.
//
// Each ring stores only its band: the hole where the finer rings are is left out (four strips per
// ring), so bytes() is the real memory use. Per cell (12 bytes): lowest and highest point (a
// pothole / an overhang), point count (a confidence), and the class of the most important point,
// by priority dynamic > static > terrain (a pedestrian outranks the kerb behind them).
//
// Plain C++ (no matrix / MLX): binning ~120k points is a few ms on the CPU, and it runs on iOS too.
// Coordinates are the sensor's: x forward, y left, z up, metres.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace foveated {

struct Cell {
    float zmin = std::numeric_limits<float>::infinity();
    float zmax = -std::numeric_limits<float>::infinity();
    uint16_t count = 0;        // points in the cell (saturates at 65535)
    uint8_t group = 0;         // highest-priority group seen: 0 unknown, 1 terrain, 2 static, 3 dynamic
    uint8_t cls = 0;           // fine class (e.g. SalsaNext's 0..19) of the point that set `group`
    bool empty() const { return count == 0; }
};
static_assert(sizeof(Cell) == 12, "Cell is meant to be 12 bytes");

struct RingSpec {
    float cell;                // cell size, metres
    float half_extent;         // the ring's outer edge: |x|, |y| < half_extent
};

// The brief's example: 5 cm within 10 m, growing to ~50 cm (40 cm here, the doubling) out to 100 m.
inline std::vector<RingSpec> default_rings() {
    return {{0.05f, 10.0f}, {0.10f, 20.0f}, {0.20f, 40.0f}, {0.40f, 100.0f}};
}

class Grid {
public:
    struct Ring {
        float cell = 0, half = 0, inner_half = 0;
        int n = 0;             // cells per side of the ring's full square
        int band = 0;          // width in cells of the band around the hole (n/2 for ring 0: no hole)
        size_t offset = 0;     // index of the ring's first cell in `cells`
        size_t count = 0;      // cells stored for this ring
    };

    explicit Grid(const std::vector<RingSpec>& specs = default_rings()) {
        float inner = 0.0f;
        size_t offset = 0;
        for (const RingSpec& s : specs) {
            Ring r;
            r.cell = s.cell;
            r.half = s.half_extent;
            r.inner_half = inner;
            r.n = whole_cells(2.0f * s.half_extent, s.cell, "ring side");
            int hole = whole_cells(2.0f * inner, s.cell, "inner ring edge");
            if (hole >= r.n || (r.n - hole) % 2 != 0) {
                throw std::invalid_argument("foveated::Grid: each ring must be wider than the one inside it, centred on it");
            }
            r.band = hole == 0 ? r.n / 2 : (r.n - hole) / 2;
            r.offset = offset;
            r.count = (size_t)r.n * r.n - (size_t)hole * hole;
            offset += r.count;
            rings_.push_back(r);
            inner = s.half_extent;
        }
        cells_.resize(offset);
    }

    const std::vector<Ring>& rings() const { return rings_; }
    const std::vector<Cell>& cells() const { return cells_; }
    size_t cell_count() const { return cells_.size(); }
    size_t bytes() const { return cells_.size() * sizeof(Cell); }
    float extent() const { return rings_.back().half; }

    void clear() { std::fill(cells_.begin(), cells_.end(), Cell{}); }

    // Index of the cell holding (x, y), or -1 if it's outside the outer ring.
    long cell_index(float x, float y) const {
        float d = std::max(std::fabs(x), std::fabs(y));
        for (const Ring& r : rings_) {
            if (d >= r.half) continue;
            int col = std::clamp((int)std::floor((x + r.half) / r.cell), 0, r.n - 1);
            int row = std::clamp((int)std::floor((y + r.half) / r.cell), 0, r.n - 1);
            const int b = r.band, n = r.n;
            if (row >= b && row < n - b && col >= b && col < n - b) {
                // Exactly on the edge of the hole (the finer ring's outer edge, e.g. y = -10 m), up
                // to rounding: the point belongs to this ring, in the strip on the side of the edge
                // it's on - the bottom/top strip if it's on a horizontal edge, left/right otherwise.
                if (std::fabs(y) >= std::fabs(x)) row = y < 0 ? b - 1 : n - b;
                else col = x < 0 ? b - 1 : n - b;
            }
            return (long)(r.offset + local_index(r, row, col));
        }
        return -1;
    }

    // Bins points into the grid. `group`/`cls` per point (may be null: then unknown / 0).
    // Returns how many points fell inside the grid.
    size_t insert(const float* xyz, size_t n, const uint8_t* group = nullptr, const uint8_t* cls = nullptr) {
        size_t inside = 0;
        for (size_t i = 0; i < n; ++i) {
            long c = cell_index(xyz[i * 3], xyz[i * 3 + 1]);
            if (c < 0) continue;
            ++inside;
            Cell& cell = cells_[(size_t)c];
            float z = xyz[i * 3 + 2];
            cell.zmin = std::min(cell.zmin, z);
            cell.zmax = std::max(cell.zmax, z);
            if (cell.count < 65535) ++cell.count;
            uint8_t g = group ? group[i] : 0;
            if (cell.count == 1 || g > cell.group) {
                cell.group = g;
                cell.cls = cls ? cls[i] : 0;
            }
        }
        return inside;
    }

    // Square covered by cell `index`: its min corner (x0, y0) and side, in metres.
    void cell_bounds(size_t index, float& x0, float& y0, float& side) const {
        for (const Ring& r : rings_) {
            if (index >= r.offset + r.count) continue;
            size_t k = index - r.offset;
            int row, col;
            local_to_row_col(r, k, row, col);
            side = r.cell;
            x0 = -r.half + col * r.cell;
            y0 = -r.half + row * r.cell;
            return;
        }
        throw std::out_of_range("foveated::Grid::cell_bounds");
    }

    size_t occupied() const {
        return (size_t)std::count_if(cells_.begin(), cells_.end(), [](const Cell& c) { return !c.empty(); });
    }

    // The same area as one uniform grid of the finest cells, for comparison.
    size_t uniform_cell_count() const {
        size_t side = (size_t)std::llround(2.0 * rings_.back().half / rings_.front().cell);
        return side * side;
    }

private:
    std::vector<Ring> rings_;
    std::vector<Cell> cells_;

    static int whole_cells(float length, float cell, const char* what) {
        double k = (double)length / cell;
        long r = std::lround(k);
        if (std::fabs(k - r) > 1e-3) {
            throw std::invalid_argument(std::string("foveated::Grid: ") + what + " isn't a whole number of cells");
        }
        return (int)r;
    }

    // A ring's band as four strips: `band` full rows at the bottom, `band` full rows at the top,
    // and in the rows between, `band` columns on the left and `band` on the right.
    static size_t local_index(const Ring& r, int row, int col) {
        const int b = r.band, n = r.n;
        if (row < b) return (size_t)row * n + col;
        if (row >= n - b) return (size_t)b * n + (size_t)(row - (n - b)) * n + col;
        size_t mid = (size_t)(row - b) * 2 * b + (col < b ? col : col - (n - b) + b);
        return (size_t)2 * b * n + mid;
    }
    static void local_to_row_col(const Ring& r, size_t k, int& row, int& col) {
        const size_t b = r.band, n = r.n;
        if (k < b * n) { row = (int)(k / n); col = (int)(k % n); return; }
        k -= b * n;
        if (k < b * n) { row = (int)(n - b + k / n); col = (int)(k % n); return; }
        k -= b * n;
        row = (int)(b + k / (2 * b));
        size_t c = k % (2 * b);
        col = (int)(c < b ? c : n - b + (c - b));
    }
};

// ---- drawing: one flat tile per occupied cell, at the cell's highest point -----------------------

struct Mesh {
    std::vector<float> xyz;        // 4 corners per tile, sensor coordinates
    std::vector<float> rgba;       // per corner
    std::vector<uint32_t> indices; // 2 triangles per tile
};

// `colour(cell, rgba_out)` picks each tile's colour. `gap` shrinks tiles a little so the cell
// structure (5 cm near, 40 cm far) stays visible.
template <typename ColourFn>
Mesh tiles(const Grid& grid, ColourFn colour, float gap = 0.1f) {
    Mesh m;
    const auto& cells = grid.cells();
    size_t occupied = grid.occupied();
    m.xyz.reserve(occupied * 12);
    m.rgba.reserve(occupied * 16);
    m.indices.reserve(occupied * 6);
    for (size_t i = 0; i < cells.size(); ++i) {
        const Cell& c = cells[i];
        if (c.empty()) continue;
        float x0, y0, s;
        grid.cell_bounds(i, x0, y0, s);
        float inset = s * gap * 0.5f;
        float xa = x0 + inset, xb = x0 + s - inset, ya = y0 + inset, yb = y0 + s - inset, z = c.zmax;
        uint32_t base = (uint32_t)(m.xyz.size() / 3);
        float corners[12] = {xa, ya, z, xb, ya, z, xb, yb, z, xa, yb, z};
        m.xyz.insert(m.xyz.end(), corners, corners + 12);
        float rgba[4];
        colour(c, rgba);
        for (int k = 0; k < 4; ++k) m.rgba.insert(m.rgba.end(), rgba, rgba + 4);
        uint32_t tri[6] = {base, base + 1, base + 2, base, base + 2, base + 3};
        m.indices.insert(m.indices.end(), tri, tri + 6);
    }
    return m;
}

} // namespace foveated

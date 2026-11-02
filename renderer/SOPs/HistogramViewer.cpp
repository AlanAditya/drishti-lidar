#pragma once
#include "GraphViewer.cpp"
#include "../matrix/matrix.h"
#include <algorithm>
#include <optional>

// A histogram as numpy returns it: counts[i] covers [edges[i], edges[i + 1]).
struct Histogram {
    matrix counts;   // [B]: Int32 counts, or Float when weighted or a density
    matrix edges;    // [B + 1] Float
};

// A bar chart's colour: one rgba for every bar, as {r, g, b, a} or a simd_float4, or a matrix:
// [B, 4] one rgba per bar, or [4] one for all (any dtype). A type rather than overloads, so a braced
// {r, g, b, a} keeps working (clang can't pick a simd_float4 overload for a braced list).
struct BarColors {
    simd_float4 one = {1, 1, 1, 1};
    std::optional<matrix> per_bar;

    BarColors(std::initializer_list<float> rgba) {
        if (rgba.size() != 4) throw std::invalid_argument("BarColors: {r, g, b, a} needs 4 values");
        auto it = rgba.begin();
        one = simd_make_float4(it[0], it[1], it[2], it[3]);
    }
    BarColors(simd_float4 rgba) : one(rgba) {}
    BarColors(const matrix& colors) { per_bar.emplace(colors); }
};

// Owns one bar chart: a mesh of B quads, one per bin, rebuilt by update(), and an optional outline
// around every bar (set_outline()), a child node drawn as a mitered line.
class BarsController {
public:
    GeoNodeImpl node;
    GeoNodeImpl outline;   // child of node: every bar's border, hidden until set_outline()
    float rwidth = 1.0f;   // fraction of its bin each bar fills (matplotlib's rwidth); < 1 leaves gaps
    // Per-bar colours [B, 4] from set_colors(), re-applied by update(); empty: one colour for every bar.
    // Optional, not a matrix member reassigned in place: the caller's matrix may be a view, and assigning
    // over a view copies into it.
    std::optional<matrix> bar_colors;
    size_m bar_count = 0;   // bars in the last update()

    BarsController(std::string name = "Bars", float rwidth = 1.0f) : rwidth(rwidth) {
        node = GeoNode::create(name);
        node->material.pipeline_state = 3;
        outline = GeoNode::create(name + " outline");
        outline->material.pipeline_state = 4;   // mitered strip: square corners
        // Coplanar with the bars, which are drawn first: pulled towards the camera so it isn't
        // lost to the depth test. Kept small: the graph's backing quad is only 0.01 behind, a few
        // steps of a 16-bit depth buffer, and a bigger bias showed the outline through it from behind.
        outline->material.depth_bias = {true, -2.0f, 0.0f, 0.0f};
        outline->visible = false;
        outline->draggable = false;
        node->add_child(outline);
    }

    // Bar i spans [edges[i], edges[i + 1]] and rises from 0 to heights[i]. Any dtype.
    void update(const matrix& edges_in, const matrix& heights_in) {
        size_m B = heights_in.shape()[0];
        if (B == 0) { clear(); return; }
        if (bar_colors && bar_colors->shape()[0] != B)
            throw std::invalid_argument("BarsController::update: " + std::to_string(B) + " bars but set_colors() gave " +
                                        std::to_string(bar_colors->shape()[0]) + " colours");
        bar_count = B;
        matrix edges = edges_in.type == dtype::Float ? edges_in : edges_in.astype(dtype::Float);
        matrix h = heights_in.type == dtype::Float ? heights_in : heights_in.astype(dtype::Float);

        // Slices are views: assigning to one copies into edges, so the narrowed bars get new names.
        matrix bin_left = edges.slice(R(0, B), 0);
        matrix bin_right = edges.slice(R(1, B + 1), 0);
        matrix centre = (bin_left + bin_right) * 0.5f;
        matrix half = (bin_right - bin_left) * (0.5f * std::clamp(rwidth, 0.0f, 1.0f));
        matrix left = centre - half;
        matrix right = centre + half;

        // 4 corners per bar (bottom-left, bottom-right, top-right, top-left), as MeshPrimitives::quad.
        matrix zero = h.zeros();
        matrix xs = matrix::stack({left, right, right, left}, -1);   // [B, 4]
        matrix ys = matrix::stack({zero, zero, h, h}, -1);           // [B, 4]
        matrix points = matrix::stack({xs, ys, xs.zeros()}, -1).reshape(4 * B, (size_m)3);

        // Outline: each bar is a closed loop bl, br, tr, tl, bl, br. Going round 1 1/4 times makes the
        // closing corner a middle point of the strip, so it gets mitered like the others (an end
        // would be cut square, leaving a notch); the repeated segment overlaps in the same colour.
        // A NaN point after each bar lifts the pen. An empty bar's loop shrinks to its centre, a
        // zero-length line the shader draws nothing for: no stray dashes along the axis. (nonzero
        // is 1 for any height, 0 for none: no comparison ops, and no NaN arithmetic, which Metal's
        // fast math doesn't promise.)
        matrix nan = zero + NAN;
        matrix nonzero = (matrix::abs(h) * 1e20f).clamp(0.0, 1.0);
        matrix outline_left = centre - half * nonzero;
        matrix outline_right = centre + half * nonzero;
        matrix loop_x = matrix::stack({outline_left, outline_right, outline_right, outline_left, outline_left, outline_right, nan}, -1);   // [B, 7]
        matrix loop_y = matrix::stack({zero, zero, h, h, zero, zero, nan}, -1);
        matrix loop = matrix::stack({loop_x, loop_y, loop_x.zeros()}, -1).reshape(7 * B, (size_m)3);

        // Two triangles per bar, offset by 4 vertices per bar.
        matrix first_vertex = matrix::linespace<int>(0, 4 * ((int)B - 1), B).unsqueeze(-1);   // [B, 1]
        matrix indices = (first_vertex + matrix::of<int32_t>({0, 1, 2, 0, 2, 3})).reshape(6 * B).astype(dtype::UInt32);
        matrix uvs = matrix::zeros({4 * B, 2}, dtype::Float);

        // Swapped in on the main queue, like LineController: the renderer reads the mesh there.
        GeoNodeImpl node_to_update = node;
        GeoNodeImpl outline_to_update = outline;
        std::optional<matrix> colors;
        if (bar_colors) colors.emplace(per_vertex(*bar_colors));
        dispatch_async(dispatch_get_main_queue(), ^{
            node_to_update->mesh.vert_position = points;
            node_to_update->mesh.indices = indices;
            node_to_update->mesh.uv_coords = uvs;
            if (colors) node_to_update->material.colors = *colors;
            outline_to_update->mesh.vert_position = loop;
            outline_to_update->mesh.indices = matrix(0, 0, dtype::UInt32);   // the vertex shader builds the line
        });
    }

    void update(const Histogram& h) { update(h.edges, h.counts); }

    // One colour per bar: [B, 4] rgba (B = the bar count), any dtype, e.g. a colormap of the heights.
    // A single [4] / [1, 4] colour colours every bar, like set_color(). Later update()s keep them, and
    // must keep the same bar count. Returns *this, so: auto bars = h->hist(x, 50).set_colors(c);
    BarsController& set_colors(const matrix& colors_in) {
        matrix c = colors_in.type == dtype::Float ? colors_in : colors_in.astype(dtype::Float);
        if (c.total_size == 4) {
            bar_colors.reset();
            GeoNodeImpl node_to_update = node;
            dispatch_async(dispatch_get_main_queue(), ^{ node_to_update->material.colors = c; });
            return *this;
        }
        if (c.dims != 2 || c.shape()[1] != 4)
            throw std::invalid_argument("BarsController::set_colors: expected [B, 4] rgba, or one [4] colour");
        if (bar_count != 0 && c.shape()[0] != bar_count)
            throw std::invalid_argument("BarsController::set_colors: " + std::to_string(c.shape()[0]) + " colours for " +
                                        std::to_string(bar_count) + " bars");
        bar_colors.reset();
        bar_colors.emplace(c);
        if (bar_count != 0) {
            matrix colors = per_vertex(c);
            GeoNodeImpl node_to_update = node;
            dispatch_async(dispatch_get_main_queue(), ^{ node_to_update->material.colors = colors; });
        }
        return *this;
    }

    // A border around every bar, like 3Blue1Brown's bar charts: colour and width (a fraction of the
    // distance to the camera, so a constant width on screen, as LineController's). Survives update().
    BarsController& set_outline(simd_float4 color = {1, 1, 1, 1}, float width = 0.0012f) {
        GeoNodeImpl outline_to_update = outline;
        dispatch_async(dispatch_get_main_queue(), ^{
            outline_to_update->material.colors = {color.x, color.y, color.z, color.w};
            outline_to_update->material.line_width = width;
            outline_to_update->visible = true;
        });
        return *this;
    }

    BarsController& remove_outline() {
        GeoNodeImpl outline_to_update = outline;
        dispatch_async(dispatch_get_main_queue(), ^{ outline_to_update->visible = false; });
        return *this;
    }

    void set_color(simd_float4 color) {
        bar_colors.reset();
        GeoNodeImpl node_to_update = node;
        dispatch_async(dispatch_get_main_queue(), ^{
            node_to_update->material.colors = {color.x, color.y, color.z, color.w};
        });
    }

    void clear() {
        GeoNodeImpl node_to_update = node;
        dispatch_async(dispatch_get_main_queue(), ^{
            node_to_update->mesh.vert_position = matrix(0, 0, dtype::Float);
            node_to_update->mesh.indices = matrix(0, 0, dtype::UInt32);
            node_to_update->mesh.uv_coords = matrix(0, 0, dtype::Float);
        });
        GeoNodeImpl outline_to_update = outline;
        dispatch_async(dispatch_get_main_queue(), ^{
            outline_to_update->mesh.vert_position = matrix(0, 0, dtype::Float);
        });
    }

private:
    // The renderer reads one colour per vertex: each bar's colour on its 4 corners, [B, 4] -> [4B, 4].
    static matrix per_vertex(const matrix& c) {
        size_m B = c.shape()[0];
        return matrix::stack({c, c, c, c}, 1).reshape(4 * B, (size_m)4);
    }
};

// A GraphViewer that draws bar charts. Counting is numpy's (bincount / histogram) and runs as
// a scatter-add, so it stays lazy and on the GPU; the viewer only draws the result.
//
//   auto [h] = r.subplots<HistogramViewer>(1, 1);
//   h->hist_int(dice_sums);                 // one bar per integer value (np.bincount)
//   h->hist(samples, 50);                   // 50 equal bins over the data's range
//   h->hist_range(samples, 50, -3, 3);      // ... over [-3, 3]; values outside are dropped
//   h->bar(edges, heights);                 // already counted
//   h->bar(edges, heights, c);              // c: [B, 4], one rgba per bar (or {r, g, b, a} for all)
//   h->hist(samples, 50).set_colors(c);     // ... or set afterwards
//   h->bar(edges, heights, c).set_outline(col::WHITE);   // a border around every bar
//
// The drawing methods take a BarColors, not overloads: clang can't convert a braced list to
// simd_float4 during overload resolution.
class HistogramViewer : public GraphViewer {
public:
    using GraphViewer::GraphViewer;

    // --- Counting (numpy-style, no drawing) ---

    // np.bincount: counts[v] = how many times v appears in x, for v in [0, max(x)] (at least
    // min_length bins). x is cast to Int32; negative values are dropped (numpy throws).
    // With weights (same shape as x), each value adds its weight instead of 1 (Float counts).
    // Reads max(x) back to size the result, so it evaluates x.
    static matrix bincount(const matrix& x_in, size_m min_length = 0, const matrix* weights = nullptr) {
        matrix x = flat(x_in.type == dtype::Int32 ? x_in : x_in.astype(dtype::Int32));
        size_m n = std::max<size_m>(min_length, (size_m)std::max(x.max().at<int32_t>() + 1, 0));
        // Shift by one so negatives land in slot 0 and get sliced off: scatter wraps negative indices.
        matrix idx = x.clamp(-1.0, (double)n) + 1;
        return count(idx, n, weights);
    }

    // np.histogram with `bins` equal-width bins over [lo, hi]. Like numpy, every bin is
    // half-open [a, b) except the last, which also takes x == hi; values outside are dropped.
    // density: counts / (total * bin width), so the bars integrate to 1.
    static Histogram histogram(const matrix& x_in, int bins, float lo, float hi,
                               const matrix* weights = nullptr, bool density = false) {
        if (bins < 1) throw std::invalid_argument("histogram: bins must be >= 1");
        if (!(hi > lo)) { lo -= 0.5f; hi += 0.5f; }   // numpy's widening for an empty/degenerate range
        matrix x = flat(x_in.type == dtype::Float ? x_in : x_in.astype(dtype::Float));

        // u: position in bins, shifted by one and clamped to [0, bins + 2] so values below lo
        // land in slot 0 and above hi in slots bins + 1 / bins + 2 (all sliced off by count()).
        // Divide before multiplying by bins so x == hi gives exactly bins + 1.
        matrix u = ((x - lo) / (hi - lo) * (float)bins).clamp(-1.0, bins + 1.0) + 1.0f;
        matrix idx = u.astype(dtype::Int32);   // floor, since u >= 0
        // Close the last bin: x == hi (u == bins + 1) goes into slot bins. The clamp/truncate pair is
        // 1 exactly when u >= bins + 1 (resp. <= bins + 1); there are no comparison ops to use.
        matrix u_at_least_edge = (u - (float)bins).clamp(0.0, 1.0).astype(dtype::Int32);
        matrix u_at_most_edge = ((bins + 2.0f) - u).clamp(0.0, 1.0).astype(dtype::Int32);
        idx = idx - u_at_least_edge * u_at_most_edge;

        // count() returns a slice (a view): assigning to it would copy into it, so build a new one.
        matrix counts = count(idx, bins, weights);
        matrix edges = matrix::linespace(lo, hi, bins + 1);
        if (!density) return {counts, edges};
        matrix c = counts.astype(dtype::Float);
        return {c / (c.sum() * ((hi - lo) / bins)), edges};
    }

    // Same, over the data's own range [min(x), max(x)] (reads both back, so it evaluates x).
    // No weights/density here: histogram(x, b, 0, 10) would be ambiguous with them (0 -> pointer).
    static Histogram histogram(const matrix& x_in, int bins = 10) {
        matrix x = x_in.type == dtype::Float ? x_in : x_in.astype(dtype::Float);
        return histogram(x, bins, x.min().at<float>(), x.max().at<float>());
    }

    // --- Drawing (counts, then draws; each returns the BarsController that owns the bars) ---

    // One bar per integer value, centred on it (bar v spans [v - 0.5, v + 0.5]).
    BarsController hist_int(const matrix& x, const BarColors& color = {1, 1, 1, 1}, float rwidth = 0.9f) {
        matrix counts = bincount(x, 0);
        size_m n = counts.shape()[0];
        return bar(matrix::linespace(-0.5f, n - 0.5f, n + 1), counts, color, rwidth);
    }

    BarsController hist(const matrix& x, int bins = 10, const BarColors& color = {1, 1, 1, 1}, float rwidth = 1.0f) {
        Histogram h = histogram(x, bins);
        return bar(h.edges, h.counts, color, rwidth);
    }

    BarsController hist_range(const matrix& x, int bins, float lo, float hi, const BarColors& color = {1, 1, 1, 1}, float rwidth = 1.0f) {
        Histogram h = histogram(x, bins, lo, hi);
        return bar(h.edges, h.counts, color, rwidth);
    }

    // Already-counted bars: edges [B + 1], heights [B], colour {r, g, b, a} or [B, 4] per bar. The first chart fits the graph to it
    // (x: the edges with a 5% margin, y: 0 up to the tallest bar + 10%) unless auto_fit is off.
    BarsController bar(const matrix& edges, const matrix& heights, const BarColors& color = {1, 1, 1, 1}, float rwidth = 1.0f) {
        if (auto_fit && plot_count == 0) {
            matrix e = edges.type == dtype::Float ? edges : edges.astype(dtype::Float);
            matrix h = heights.type == dtype::Float ? heights : heights.astype(dtype::Float);
            float xmin = e.min().at<float>(), xmax = e.max().at<float>();
            float ymax = std::max(h.max().at<float>(), 0.0f);
            float ymin = std::min(h.min().at<float>(), 0.0f);
            if (xmax <= xmin) { xmin -= 1; xmax += 1; }
            if (ymax <= ymin) { ymax = ymin + 1; }
            float mx = 0.05f * (xmax - xmin), my = 0.1f * (ymax - ymin);
            set_bounds({xmin - mx, xmax + mx, ymin - (ymin < 0 ? my : 0.0f), ymax + my});
        }
        plot_count++;
        BarsController bars("bars", rwidth);
        bars.node->material.colors = {color.one.x, color.one.y, color.one.z, color.one.w};
        if (color.per_bar) bars.set_colors(*color.per_bar);   // stored; the update() below applies it
        add_child(bars.node, simd_make_float3(0, 0, 0));
        bars.outline->material.clip_min = bars.node->material.clip_min;   // clipped to the frame like the bars
        bars.outline->material.clip_max = bars.node->material.clip_max;
        bars.update(edges, heights);
        return bars;
    }

private:
    static matrix flat(matrix x) { return x.dims == 1 ? x : x.reshape((size_m)x.total_size); }

    // Scatter-adds 1 (or the weight) into slot idx of a [bins + 3] buffer, then keeps slots 1..bins:
    // slot 0 and the ones past bins are where the callers send out-of-range values.
    static matrix count(const matrix& idx, size_m bins, const matrix* weights) {
        size_m n = idx.shape()[0];
        matrix updates = weights ? flat(weights->type == dtype::Float ? *weights : weights->astype(dtype::Float))
                                 : matrix::ones({n}, dtype::Int32);
        dtype type = weights ? dtype::Float : dtype::Int32;
        matrix slots = matrix::scatter(matrix::zeros({bins + 3}, type), idx, updates, 0, ScatterMode::ADD);
        return slots.slice(R(1, bins + 1), 0);
    }
};

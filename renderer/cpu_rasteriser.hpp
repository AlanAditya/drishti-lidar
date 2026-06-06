//
//  cpu_rasteriser.hpp
//  AdityaIntelligenceProMax
//

#ifndef cpu_rasteriser_hpp
#define cpu_rasteriser_hpp

#include <algorithm>
#include <cmath>
#import <CoreText/CoreText.h>
#import <CoreGraphics/CoreGraphics.h>
#import <simd/simd.h>

// Forward declaration; the compilation unit including this MUST have the full `matrix` definition beforehand.
class matrix;

enum class PathVerb : uint8_t {
    MoveTo,
    LineTo,
    QuadTo,
    CubicTo,
    Close
};

struct Path2D {
    std::vector<PathVerb> verbs;
    std::vector<float> _temp_points;
    matrix points;
    bool finalized = false;
    
    Path2D() : points(2, dtype::Float) {}

    void moveTo(float x, float y) {
        verbs.push_back(PathVerb::MoveTo);
        _temp_points.push_back(x);
        _temp_points.push_back(y);
    }
    
    void lineTo(float x, float y) {
        verbs.push_back(PathVerb::LineTo);
        _temp_points.push_back(x);
        _temp_points.push_back(y);
    }
    
    void quadTo(float cx, float cy, float x, float y) {
        verbs.push_back(PathVerb::QuadTo);
        _temp_points.push_back(cx);
        _temp_points.push_back(cy);
        _temp_points.push_back(x);
        _temp_points.push_back(y);
    }
    
    void cubicTo(float cx1, float cy1, float cx2, float cy2, float x, float y) {
        verbs.push_back(PathVerb::CubicTo);
        _temp_points.push_back(cx1);
        _temp_points.push_back(cy1);
        _temp_points.push_back(cx2);
        _temp_points.push_back(cy2);
        _temp_points.push_back(x);
        _temp_points.push_back(y);
    }
    
    void close() {
        verbs.push_back(PathVerb::Close);
    }
    
    void freeze() {
        if (_temp_points.empty() || finalized) return;
        points = matrix::withShape({(size_m)(_temp_points.size() / 2), 2}, dtype::Float);
        memcpy(points.buffer, _temp_points.data(), _temp_points.size() * sizeof(float));
        std::vector<float>().swap(_temp_points);
        finalized = true;
    }
};

// --- PLOT SYSTEM ---

enum class PlotStyle : uint8_t {
    Line,
    Scatter,
    Both
};

struct PlotAxis {
    float min = 0.0f;
    float max = 1.0f;
    int tickCount = 5;
    const char* title = "";
    const char* fontName = "Helvetica";
    float fontSize = 14.0f;
    float tickFontSize = 11.0f;
};

struct PlotSeries {
    matrix x;
    matrix y;
    matrix color;
    PlotStyle style = PlotStyle::Line;
    float lineWidth = 2.0f;
    float pointRadius = 4.0f;
    
    PlotSeries() : x(1, dtype::Float), y(1, dtype::Float), color(1, dtype::UInt8) {}
};

struct Plot {
    PlotAxis xAxis;
    PlotAxis yAxis;
    const char* title = "";
    const char* fontName = "Helvetica";
    float titleFontSize = 20.0f;
    
    int marginLeft = 80;
    int marginRight = 30;
    int marginTop = 50;
    int marginBottom = 70;
    
    matrix bgColor;
    matrix plotAreaColor;
    matrix axisColor;
    matrix gridColor;
    
    bool showGrid = true;
    float axisLineWidth = 2.0f;
    float gridLineWidth = 1.0f;
    
    static const int MAX_SERIES = 16;
    PlotSeries series[MAX_SERIES];
    int seriesCount = 0;
    
    Plot()
      : bgColor(matrix{30u, 30u, 30u, 255u}),
        plotAreaColor(matrix{15u, 15u, 15u, 255u}),
        axisColor(matrix{200u, 200u, 200u, 255u}),
        gridColor(matrix{60u, 60u, 60u, 255u}) {}
    
    void addSeries(const matrix& x_data, const matrix& y_data, const matrix& col,
                   PlotStyle style = PlotStyle::Line, float lineW = 2.0f, float pointR = 4.0f) {
        if (seriesCount >= MAX_SERIES) return;
        PlotSeries& s = series[seriesCount];
        s.x = x_data;
        s.y = y_data;
        s.color = col;
        s.style = style;
        s.lineWidth = lineW;
        s.pointRadius = pointR;
        seriesCount++;
    }
};
class CPURasteriser {
public:
    // Takes thickness and two points that the line needs to be drawn on
    static void drawLine(matrix& canvas, int x0, int y0, int x1, int y1, float thickness, const matrix& fill) {
        if (canvas.dims < 2) return;
        
        int h = canvas.shape()[0];
        int w = canvas.shape()[1];
        
        int expand = static_cast<int>(std::ceil(thickness / 2.0f + 0.5f));
        int minX = std::max(0, std::min(x0, x1) - expand);
        int maxX = std::min(w - 1, std::max(x0, x1) + expand);
        int minY = std::max(0, std::min(y0, y1) - expand);
        int maxY = std::min(h - 1, std::max(y0, y1) + expand);
        
        int channels = (canvas.dims >= 3) ? canvas.shape()[2] : 1;
        uint8_t* canvas_data = static_cast<uint8_t*>(canvas.buffer);
        float half_thickness = thickness / 2.0f;
        
        // Convert to UInt8 array since canvas buffer expects Uint8 matching bytes natively.
        // We do this to manually ensure there are no use-after-free or copy construction ownership leaks
        // that can sometimes happen when passing temps into in-place casting functions over C++ memory boundaries.
        matrix fill_uint8 = matrix(fill.dims, dtype::UInt8);
        memcpy(fill_uint8.shape(), fill.shape(), sizeof(size_m) * fill.dims);
        memcpy(fill_uint8.strides(), fill.strides(), sizeof(size_m) * fill.dims);
        fill_uint8.total_size = fill.total_size;
        fill_uint8.type = dtype::UInt8;
        fill_uint8.buffer = malloc(fill_uint8.total_size);
        matrix::copyCPUinplaceTypeCasted(fill_uint8, fill, 0);
        
        uint8_t fill_data[32] = {}; // Handle up to 32 channels safely
        memcpy(fill_data, fill_uint8.buffer, sizeof(uint8_t) * fill_uint8.total_size);
        
        float expand_f = half_thickness + 0.5f;
        float pad_sq = expand_f * expand_f;
        
        float dx = x1 - x0;
        float dy = y1 - y0;
        float l2 = dx * dx + dy * dy;
        size_m* canvas_strides = canvas.strides();
        // Blit across the rasterized distances natively without sqrt!
        for (int y = minY; y <= maxY; ++y) {
            float py_y0 = y - y0;
            for (int x = minX; x <= maxX; ++x) {
                float px_x0 = x - x0;
                float d_sq;
                
                if (l2 == 0.0f) {
                    d_sq = px_x0 * px_x0 + py_y0 * py_y0;
                } else {
                    float t = (px_x0 * dx + py_y0 * dy) / l2;
                    t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
                    float projX = x0 + t * dx;
                    float projY = y0 + t * dy;
                    float dpx = x - projX;
                    float dpy = y - projY;
                    d_sq = dpx * dpx + dpy * dpy;
                }
                
                if (d_sq <= pad_sq) {
                    float d = std::sqrt(d_sq);
                    float a = std::max(0.0f, std::min(1.0f, half_thickness - d + 0.5f));
                    
                    if (a > 0.0f) {
                        int pixel_idx = y * canvas_strides[0] + x * canvas_strides[1];
                        if (a >= 0.99f) {
                            for (int c = 0; c < channels; ++c) {
                                int c_offset = (canvas.dims >= 3) ? c * canvas_strides[2] : 0;
                                canvas_data[pixel_idx + c_offset] = fill_data[c];
                            }
                        } else {
                            for (int c = 0; c < channels; ++c) {
                                int c_offset = (canvas.dims >= 3) ? c * canvas_strides[2] : 0;
                                canvas_data[pixel_idx + c_offset] = static_cast<uint8_t>(
                                    canvas_data[pixel_idx + c_offset] * (1.0f - a) + fill_data[c] * a
                                );
                            }
                        }
                    }
                }
            }
        }
        
    }

    // High-level wrapper that uses matrix slicing and assignment
    // (Caution: Significantly slower due to temporary matrix instantiations in the loop)
    static void drawLineGG(matrix& canvas, int x0, int y0, int x1, int y1, float thickness, const matrix& fill) {
        if (canvas.dims < 2) return;
        
        int h = canvas.shape()[0];
        int w = canvas.shape()[1];
        
        int minX = std::max(0, std::min(x0, x1) - (int)std::ceil(thickness));
        int maxX = std::min(w - 1, std::max(x0, x1) + (int)std::ceil(thickness));
        int minY = std::max(0, std::min(y0, y1) - (int)std::ceil(thickness));
        int maxY = std::min(h - 1, std::max(y0, y1) + (int)std::ceil(thickness));
        

        matrix fill_uint8 = matrix(fill.dims, dtype::UInt8);
        memcpy(fill_uint8.shape(), fill.shape(), sizeof(size_m) * fill.dims);
        memcpy(fill_uint8.strides(), fill.strides(), sizeof(size_m) * fill.dims);
        fill_uint8.total_size = fill.total_size;
        fill_uint8.type = dtype::UInt8;
        fill_uint8.buffer = malloc(fill_uint8.total_size);
        matrix::copyCPUinplaceTypeCasted(fill_uint8, fill, 0);
        
        float half_thickness_sq = (thickness * thickness) / 4.0f;
        
        float dx = x1 - x0;
        float dy = y1 - y0;
        float l2 = dx * dx + dy * dy;
        
        for (int y = minY; y <= maxY; ++y) {
            float py_y0 = y - y0;
            for (int x = minX; x <= maxX; ++x) {
                float px_x0 = x - x0;
                float d_sq;
                
                if (l2 == 0.0f) {
                    d_sq = px_x0 * px_x0 + py_y0 * py_y0;
                } else {
                    float t = (px_x0 * dx + py_y0 * dy) / l2;
                    t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
                    float projX = x0 + t * dx;
                    float projY = y0 + t * dy;
                    float dpx = x - projX;
                    float dpy = y - projY;
                    d_sq = dpx * dpx + dpy * dpy;
                }
                
                if (d_sq <= half_thickness_sq) {
                    // Use the custom native multidimensional subscript operator!
                    // This creates a (dims-2) view representing the pixel channels, and invokes copyCPUinplace under the hood!
                    canvas[y, x] = fill_uint8;
                }
            }
        }
    }

    // version 1: takes in points as from x:[-1,1] y:[-1,1] and maps them to matrix index
    static void drawLine(matrix& canvas, float nx0, float ny0, float nx1, float ny1, float thickness, const matrix& fill) {
        if (canvas.dims < 2) return;
        
        int h = canvas.shape()[0];
        int w = canvas.shape()[1];
        
        int idx_x0 = static_cast<int>(std::round((nx0 + 1.0f) * 0.5f * (w - 1)));
        int idx_y0 = static_cast<int>(std::round((1.0f - ny0) * 0.5f * (h - 1)));
        
        int idx_x1 = static_cast<int>(std::round((nx1 + 1.0f) * 0.5f * (w - 1)));
        int idx_y1 = static_cast<int>(std::round((1.0f - ny1) * 0.5f * (h - 1)));
        
        drawLine(canvas, idx_x0, idx_y0, idx_x1, idx_y1, thickness, fill);
    }
    
    // --- RECTANGLE DRAWING ---
    
    // Draws a solid filled rectangle directly utilizing slicing broadcasting!
    static void drawRect(matrix& canvas, int x0, int y0, int x1, int y1, const matrix& fill) {
        if (canvas.dims < 2) return;
        
        int w = canvas.shape()[1];
        int h = canvas.shape()[0];
        
        int minX = std::max(0, std::min(x0, x1));
        int maxX = std::min(w - 1, std::max(x0, x1));
        int minY = std::max(0, std::min(y0, y1));
        int maxY = std::min(h - 1, std::max(y0, y1));
        
        if (minX > maxX || minY > maxY) return;
        
        // Natively slice the exact boundary. Assignment operator will auto-broadcast the fill
        // safely and execute a CPU-side matrix fill taking advantage of identical memory blocks.
        canvas.slice({R(minY, maxY + 1), R(minX, maxX + 1)}) = fill;
    }
    
    static void drawRect(matrix& canvas, float nx0, float ny0, float nx1, float ny1, const matrix& fill) {
        if (canvas.dims < 2) return;
        int w = canvas.shape()[1];
        int h = canvas.shape()[0];
        int idx_x0 = static_cast<int>(std::round((nx0 + 1.0f) * 0.5f * (w - 1)));
        int idx_y0 = static_cast<int>(std::round((1.0f - ny0) * 0.5f * (h - 1)));
        int idx_x1 = static_cast<int>(std::round((nx1 + 1.0f) * 0.5f * (w - 1)));
        int idx_y1 = static_cast<int>(std::round((1.0f - ny1) * 0.5f * (h - 1)));
        drawRect(canvas, idx_x0, idx_y0, idx_x1, idx_y1, fill);
    }
    
    // --- CIRCLE DRAWING ---
    
    // Draws a solid filled circle using fast Horizontal Scanlines!
    static void drawCircle(matrix& canvas, int cx, int cy, float radius, const matrix& fill) {
        if (canvas.dims < 2 || radius <= 0.0f) return;
        
        int w = canvas.shape()[1];
        int h = canvas.shape()[0];
        
        int minY = std::max(0, static_cast<int>(std::ceil(cy - radius)));
        int maxY = std::min(h - 1, static_cast<int>(std::floor(cy + radius)));
        
        float r_sq = radius * radius;
        
        matrix fill_uint8 = matrix(fill.dims, dtype::UInt8);
        memcpy(fill_uint8.shape(), fill.shape(), sizeof(size_m) * fill.dims);
        memcpy(fill_uint8.strides(), fill.strides(), sizeof(size_m) * fill.dims);
        fill_uint8.total_size = fill.total_size;
        fill_uint8.type = dtype::UInt8;
        fill_uint8.buffer = malloc(fill_uint8.total_size);
        matrix::copyCPUinplaceTypeCasted(fill_uint8, fill, 0);
        
        // Scanline logic ensures we only modify pixels inside the circle.
        // Bypasses the danger of a direct top-to-bottom rectangle `memcpy` wiping out the background canvas details outside the circle radius!
        for (int y = minY; y <= maxY; ++y) {
            float dy = static_cast<float>(y - cy);
            float dy_sq = dy * dy;
            
            if (dy_sq > r_sq) continue;
            
            float dx = std::sqrt(r_sq - dy_sq);
            
            int minX = std::max(0, static_cast<int>(std::ceil(cx - dx)));
            int maxX = std::min(w - 1, static_cast<int>(std::floor(cx + dx)));
            
            if (minX <= maxX) {
                // Paint this specific horizontal horizontal slice perfectly
                canvas.slice({R(y, y + 1), R(minX, maxX + 1)}) = fill_uint8;
            }
        }
    }
    
    static void drawCircle(matrix& canvas, float nx, float ny, float nradius, const matrix& fill) {
        if (canvas.dims < 2) return;
        int w = canvas.shape()[1];
        int h = canvas.shape()[0];
        
        int cx = static_cast<int>(std::round((nx + 1.0f) * 0.5f * (w - 1)));
        int cy = static_cast<int>(std::round((1.0f - ny) * 0.5f * (h - 1)));
        // Approximate pixel radius using horizontal axis resolution (assumes square aspect ratio mapping natively in tensor forms)
        float r = nradius * 0.5f * (w - 1);
        
        drawCircle(canvas, cx, cy, r, fill);
    }

    // --- STROKED VARIANTS ---
    
    // Draws just the stroked edges of a rectangle
    static void drawStrokedRect(matrix& canvas, int x0, int y0, int x1, int y1, float stroke_width, const matrix& fill) {
        if (canvas.dims < 2) return;
        
        int w = canvas.shape()[1];
        int h = canvas.shape()[0];
        
        int minX = std::max(0, std::min(x0, x1));
        int maxX = std::min(w - 1, std::max(x0, x1));
        int minY = std::max(0, std::min(y0, y1));
        int maxY = std::min(h - 1, std::max(y0, y1));
        
        if (minX > maxX || minY > maxY) return;
        
        int thickness = std::max(1, static_cast<int>(std::round(stroke_width)));
        
        // Top Edge
        int top_end = std::min(maxY + 1, minY + thickness);
        if (minY < top_end) canvas.slice({R(minY, top_end), R(minX, maxX + 1)}) = fill;
        
        // Bottom Edge
        int bot_start = std::max(minY, maxY + 1 - thickness);
        if (bot_start < maxY + 1) canvas.slice({R(bot_start, maxY + 1), R(minX, maxX + 1)}) = fill;
        
        // Left Edge
        int left_end = std::min(maxX + 1, minX + thickness);
        int mid_minY = std::min(top_end, maxY + 1);
        int mid_maxY = std::max(bot_start, minY);
        if (mid_minY < mid_maxY && minX < left_end) {
            canvas.slice({R(mid_minY, mid_maxY), R(minX, left_end)}) = fill;
        }
        
        // Right Edge
        int right_start = std::max(minX, maxX + 1 - thickness);
        if (mid_minY < mid_maxY && right_start < maxX + 1) {
            canvas.slice({R(mid_minY, mid_maxY), R(right_start, maxX + 1)}) = fill;
        }
    }
    
    static void drawStrokedRect(matrix& canvas, float nx0, float ny0, float nx1, float ny1, float nstroke_width, const matrix& fill) {
        if (canvas.dims < 2) return;
        int w = canvas.shape()[1];
        int h = canvas.shape()[0];
        int idx_x0 = static_cast<int>(std::round((nx0 + 1.0f) * 0.5f * (w - 1)));
        int idx_y0 = static_cast<int>(std::round((1.0f - ny0) * 0.5f * (h - 1)));
        int idx_x1 = static_cast<int>(std::round((nx1 + 1.0f) * 0.5f * (w - 1)));
        int idx_y1 = static_cast<int>(std::round((1.0f - ny1) * 0.5f * (h - 1)));
        float s_width = std::max(1.0f, nstroke_width * 0.5f * (w - 1));
        drawStrokedRect(canvas, idx_x0, idx_y0, idx_x1, idx_y1, s_width, fill);
    }
    
    // Draws just the stroked edges of a circle
    static void drawStrokedCircle(matrix& canvas, int cx, int cy, float radius, float stroke_width, const matrix& fill) {
        if (canvas.dims < 2 || radius <= 0.0f) return;
        
        int w = canvas.shape()[1];
        int h = canvas.shape()[0];
        
        float outer_radius = radius;
        float inner_radius = std::max(0.0f, radius - stroke_width);
        
        int minY = std::max(0, static_cast<int>(std::ceil(cy - outer_radius)));
        int maxY = std::min(h - 1, static_cast<int>(std::floor(cy + outer_radius)));
        
        float outer_r_sq = outer_radius * outer_radius;
        float inner_r_sq = inner_radius * inner_radius;
        
        matrix fill_uint8 = matrix(fill.dims, dtype::UInt8);
        memcpy(fill_uint8.shape(), fill.shape(), sizeof(size_m) * fill.dims);
        memcpy(fill_uint8.strides(), fill.strides(), sizeof(size_m) * fill.dims);
        fill_uint8.total_size = fill.total_size;
        fill_uint8.type = dtype::UInt8;
        fill_uint8.buffer = malloc(fill_uint8.total_size);
        matrix::copyCPUinplaceTypeCasted(fill_uint8, fill, 0);
        
        for (int y = minY; y <= maxY; ++y) {
            float dy = static_cast<float>(y - cy);
            float dy_sq = dy * dy;
            
            if (dy_sq > outer_r_sq) continue;
            
            float dx_outer = std::sqrt(outer_r_sq - dy_sq);
            
            int outer_minX = std::max(0, static_cast<int>(std::ceil(cx - dx_outer)));
            int outer_maxX = std::min(w - 1, static_cast<int>(std::floor(cx + dx_outer)));
            
            if (outer_minX > outer_maxX) continue;
            
            if (dy_sq >= inner_r_sq) {
                // Top or bottom arches (solid)
                canvas.slice({R(y, y + 1), R(outer_minX, outer_maxX + 1)}) = fill_uint8;
            } else {
                float dx_inner = std::sqrt(inner_r_sq - dy_sq);
                int inner_minX = std::max(0, static_cast<int>(std::ceil(cx - dx_inner)));
                int inner_maxX = std::min(w - 1, static_cast<int>(std::floor(cx + dx_inner)));
                
                // Left
                if (outer_minX < inner_minX) {
                    canvas.slice({R(y, y + 1), R(outer_minX, inner_minX)}) = fill_uint8;
                }
                
                // Right
                int right_start = std::max(inner_maxX + 1, outer_minX);
                if (right_start <= outer_maxX) {
                    canvas.slice({R(y, y + 1), R(right_start, outer_maxX + 1)}) = fill_uint8;
                }
            }
        }
    }
    
    static void drawStrokedCircle(matrix& canvas, float nx, float ny, float nradius, float nstroke_width, const matrix& fill) {
        if (canvas.dims < 2) return;
        int w = canvas.shape()[1];
        int h = canvas.shape()[0];
        
        int cx = static_cast<int>(std::round((nx + 1.0f) * 0.5f * (w - 1)));
        int cy = static_cast<int>(std::round((1.0f - ny) * 0.5f * (h - 1)));
        float r = nradius * 0.5f * (w - 1);
        float s_width = std::max(1.0f, nstroke_width * 0.5f * (w - 1));
        
        drawStrokedCircle(canvas, cx, cy, r, s_width, fill);
    }
    // --- BEZIER CURVES ---
    
    // Draws a Quadratic Bezier Curve (1 control point) natively by subdividing into accelerated line segments
    static void drawQuadraticBezier(matrix& canvas, int x0, int y0, int cx, int cy, int x1, int y1, float thickness, const matrix& fill, int segments = 50) {
        if (canvas.dims < 2 || segments <= 0) return;
        
        float step = 1.0f / segments;
        float prev_x = x0;
        float prev_y = y0;
        
        for (int i = 1; i <= segments; ++i) {
            float t = i * step;
            float inv_t = 1.0f - t;
            
            // Quadratic Bezier formula
            float curr_x = inv_t * inv_t * x0 + 2.0f * inv_t * t * cx + t * t * x1;
            float curr_y = inv_t * inv_t * y0 + 2.0f * inv_t * t * cy + t * t * y1;
            
            drawLine(canvas, static_cast<int>(std::round(prev_x)), static_cast<int>(std::round(prev_y)),
                             static_cast<int>(std::round(curr_x)), static_cast<int>(std::round(curr_y)), thickness, fill);
            prev_x = curr_x;
            prev_y = curr_y;
        }
    }
    
    static void drawQuadraticBezier(matrix& canvas, float nx0, float ny0, float cx, float cy, float nx1, float ny1, float thickness, const matrix& fill, int segments = 50) {
        int w = canvas.shape()[1];
        int h = canvas.shape()[0];
        drawQuadraticBezier(canvas,
            static_cast<int>(std::round((nx0 + 1.0f) * 0.5f * (w - 1))), static_cast<int>(std::round((1.0f - ny0) * 0.5f * (h - 1))),
            static_cast<int>(std::round((cx + 1.0f) * 0.5f * (w - 1))),  static_cast<int>(std::round((1.0f - cy) * 0.5f * (h - 1))),
            static_cast<int>(std::round((nx1 + 1.0f) * 0.5f * (w - 1))), static_cast<int>(std::round((1.0f - ny1) * 0.5f * (h - 1))),
            thickness, fill, segments);
    }
    
    // Draws a Cubic Bezier Curve (2 control points)
    static void drawCubicBezier(matrix& canvas, int x0, int y0, int cx0, int cy0, int cx1, int cy1, int x1, int y1, float thickness, const matrix& fill, int segments = 50) {
        if (canvas.dims < 2 || segments <= 0) return;
        
        float step = 1.0f / segments;
        float prev_x = x0;
        float prev_y = y0;
        
        for (int i = 1; i <= segments; ++i) {
            float t = i * step;
            float inv_t = 1.0f - t;
            
            float inv_t_sq = inv_t * inv_t;
            float t_sq = t * t;
            
            // Cubic Bezier formula
            float curr_x = inv_t_sq * inv_t * x0 + 3.0f * inv_t_sq * t * cx0 + 3.0f * inv_t * t_sq * cx1 + t_sq * t * x1;
            float curr_y = inv_t_sq * inv_t * y0 + 3.0f * inv_t_sq * t * cy0 + 3.0f * inv_t * t_sq * cy1 + t_sq * t * y1;
            
            drawLine(canvas, static_cast<int>(std::round(prev_x)), static_cast<int>(std::round(prev_y)),
                             static_cast<int>(std::round(curr_x)), static_cast<int>(std::round(curr_y)), thickness, fill);
            prev_x = curr_x;
            prev_y = curr_y;
        }
    }
    
    static void drawCubicBezier(matrix& canvas, float nx0, float ny0, float cx0, float cy0, float cx1, float cy1, float nx1, float ny1, float thickness, const matrix& fill, int segments = 50) {
        int w = canvas.shape()[1];
        int h = canvas.shape()[0];
        drawCubicBezier(canvas,
            static_cast<int>(std::round((nx0 + 1.0f) * 0.5f * (w - 1))), static_cast<int>(std::round((1.0f - ny0) * 0.5f * (h - 1))),
            static_cast<int>(std::round((cx0 + 1.0f) * 0.5f * (w - 1))), static_cast<int>(std::round((1.0f - cy0) * 0.5f * (h - 1))),
            static_cast<int>(std::round((cx1 + 1.0f) * 0.5f * (w - 1))), static_cast<int>(std::round((1.0f - cy1) * 0.5f * (h - 1))),
            static_cast<int>(std::round((nx1 + 1.0f) * 0.5f * (w - 1))), static_cast<int>(std::round((1.0f - ny1) * 0.5f * (h - 1))),
            thickness, fill, segments);
    }
    
    // --- POLYGONS ---
    
    // Scanline polygon fill (Integer indices expected natively)
    static void drawPolygon(matrix& canvas, const matrix& points, const matrix& fill) {
        if (canvas.dims < 2 || points.dims != 2 || points.shape()[1] != 2 || points.shape()[0] < 3) return;
        
        matrix int_points = points;
        if (points.type != dtype::Int32) {
            int_points = matrix(points.dims, dtype::Int32);
            memcpy(int_points.shape(), points.shape(), sizeof(size_m) * points.dims);
            memcpy(int_points.strides(), points.strides(), sizeof(size_m) * points.dims);
            int_points.total_size = points.total_size;
            int_points.buffer = malloc(int_points.total_size * 4);
            matrix::copyCPUinplaceTypeCasted(int_points, points, 0);
        }
        
        int w = canvas.shape()[1];
        int h = canvas.shape()[0];
        int n = int_points.shape()[0];
        int32_t* data = static_cast<int32_t*>(int_points.buffer);
        
        int minY = h, maxY = 0;
        for (int i = 0; i < n; ++i) {
            int y = data[i * 2 + 1];
            minY = std::min(minY, y);
            maxY = std::max(maxY, y);
        }
        
        minY = std::max(0, minY);
        maxY = std::min(h - 1, maxY);
        
        if (minY > maxY) {
            if (points.type != dtype::Int32) free(int_points.buffer);
            return;
        }
        
        matrix fill_uint8 = matrix(fill.dims, dtype::UInt8);
        memcpy(fill_uint8.shape(), fill.shape(), sizeof(size_m) * fill.dims);
        memcpy(fill_uint8.strides(), fill.strides(), sizeof(size_m) * fill.dims);
        fill_uint8.total_size = fill.total_size;
        fill_uint8.type = dtype::UInt8;
        fill_uint8.buffer = malloc(fill_uint8.total_size);
        matrix::copyCPUinplaceTypeCasted(fill_uint8, fill, 0);

        for (int y = minY; y <= maxY; ++y) {
            std::vector<int> nodeX;
            for (int i = 0, j = n - 1; i < n; j = i++) {
                int yi = data[i * 2 + 1];
                int yj = data[j * 2 + 1];
                if ((yi < y && yj >= y) || (yj < y && yi >= y)) {
                    int xi = data[i * 2];
                    int xj = data[j * 2];
                    nodeX.push_back(xi + static_cast<int>(static_cast<float>(y - yi) / (yj - yi) * (xj - xi)));
                }
            }
            std::sort(nodeX.begin(), nodeX.end());
            
            for (size_t i = 0; i + 1 < nodeX.size(); i += 2) {
                int startX = std::max(0, nodeX[i]);
                int endX = std::min(w - 1, nodeX[i+1]);
                if (startX <= endX) {
                    canvas.slice({R(y, y + 1), R(startX, endX + 1)}) = fill_uint8;
                }
            }
        }
        
    }
    
    // Scanline polygon fill (Float System Normalized [-1, 1] natively)
    static void drawPolygonNormalized(matrix& canvas, const matrix& npoints, const matrix& fill) {
        if (canvas.dims < 2 || npoints.dims != 2 || npoints.shape()[1] != 2 || npoints.shape()[0] < 3) return;
        
        matrix float_points = npoints;
        if (npoints.type != dtype::Float) {
            float_points = matrix(npoints.dims, dtype::Float);
            memcpy(float_points.shape(), npoints.shape(), sizeof(size_m) * npoints.dims);
            memcpy(float_points.strides(), npoints.strides(), sizeof(size_m) * npoints.dims);
            float_points.total_size = npoints.total_size;
            float_points.buffer = malloc(float_points.total_size * 4);
            matrix::copyCPUinplaceTypeCasted(float_points, npoints, 0);
        }
        
        int w = canvas.shape()[1];
        int h = canvas.shape()[0];
        int n = float_points.shape()[0];
        float* fdata = static_cast<float*>(float_points.buffer);
        
        matrix int_points = matrix(npoints.dims, dtype::Int32);
        memcpy(int_points.shape(), npoints.shape(), sizeof(size_m) * npoints.dims);
        memcpy(int_points.strides(), npoints.strides(), sizeof(size_m) * npoints.dims);
        int_points.total_size = npoints.total_size;
        int_points.buffer = malloc(int_points.total_size * 4);
        int32_t* idata = static_cast<int32_t*>(int_points.buffer);
        
        for (int i = 0; i < n; ++i) {
            idata[i * 2]     = static_cast<int>(std::round((fdata[i * 2] + 1.0f) * 0.5f * (w - 1)));
            idata[i * 2 + 1] = static_cast<int>(std::round((1.0f - fdata[i * 2 + 1]) * 0.5f * (h - 1)));
        }
        
        drawPolygon(canvas, int_points, fill);
    }
    
    // Draw stroked polygon edges (Integer indices natively)
    static void drawStrokedPolygon(matrix& canvas, const matrix& points, float thickness, const matrix& fill, bool close = true) {
        if (canvas.dims < 2 || points.dims != 2 || points.shape()[1] != 2 || points.shape()[0] < 2) return;
        
        matrix int_points = points;
        if (points.type != dtype::Int32) {
            int_points = matrix(points.dims, dtype::Int32);
            memcpy(int_points.shape(), points.shape(), sizeof(size_m) * points.dims);
            memcpy(int_points.strides(), points.strides(), sizeof(size_m) * points.dims);
            int_points.total_size = points.total_size;
            int_points.buffer = malloc(int_points.total_size * 4);
            matrix::copyCPUinplaceTypeCasted(int_points, points, 0);
        }
        
        int n = int_points.shape()[0];
        int32_t* data = static_cast<int32_t*>(int_points.buffer);
        
        for (int i = 0; i < n - 1; ++i) {
            drawLine(canvas, data[i * 2], data[i * 2 + 1], data[(i+1) * 2], data[(i+1) * 2 + 1], thickness, fill);
        }
        
        if (close && n > 2) {
            drawLine(canvas, data[(n-1) * 2], data[(n-1) * 2 + 1], data[0], data[1], thickness, fill);
        }
        
        if (points.type != dtype::Int32) free(int_points.buffer);
    }

    // Draw stroked polygon edges (Float System Normalized [-1, 1] natively)
    static void drawStrokedPolygonNormalized(matrix& canvas, const matrix& npoints, float thickness, const matrix& fill, bool close = true) {
        if (canvas.dims < 2 || npoints.dims != 2 || npoints.shape()[1] != 2 || npoints.shape()[0] < 2) return;
        
        matrix float_points = npoints;
        if (npoints.type != dtype::Float) {
            float_points = matrix(npoints.dims, dtype::Float);
            memcpy(float_points.shape(), npoints.shape(), sizeof(size_m) * npoints.dims);
            memcpy(float_points.strides(), npoints.strides(), sizeof(size_m) * npoints.dims);
            float_points.total_size = npoints.total_size;
            float_points.buffer = malloc(float_points.total_size * 4);
            matrix::copyCPUinplaceTypeCasted(float_points, npoints, 0);
        }
        
        int w = canvas.shape()[1];
        int h = canvas.shape()[0];
        int n = float_points.shape()[0];
        float* fdata = static_cast<float*>(float_points.buffer);
        
        matrix int_points = matrix(npoints.dims, dtype::Int32);
        memcpy(int_points.shape(), npoints.shape(), sizeof(size_m) * npoints.dims);
        memcpy(int_points.strides(), npoints.strides(), sizeof(size_m) * npoints.dims);
        int_points.total_size = npoints.total_size;
        int_points.buffer = malloc(int_points.total_size * 4);
        int32_t* idata = static_cast<int32_t*>(int_points.buffer);
        
        for (int i = 0; i < n; ++i) {
            idata[i * 2]     = static_cast<int>(std::round((fdata[i * 2] + 1.0f) * 0.5f * (w - 1)));
            idata[i * 2 + 1] = static_cast<int>(std::round((1.0f - fdata[i * 2 + 1]) * 0.5f * (h - 1)));
        }
        
        drawStrokedPolygon(canvas, int_points, thickness, fill, close);
    }
    
    // --- ELLIPSES ---

    // Scanline Ellipse fill
    static void drawEllipse(matrix& canvas, int cx, int cy, float rx, float ry, const matrix& fill) {
        if (canvas.dims < 2 || rx <= 0.0f || ry <= 0.0f) return;
        
        int w = canvas.shape()[1];
        int h = canvas.shape()[0];
        
        int minY = std::max(0, static_cast<int>(std::ceil(cy - ry)));
        int maxY = std::min(h - 1, static_cast<int>(std::floor(cy + ry)));
        
        float rx_sq = rx * rx;
        float ry_sq = ry * ry;
        
        matrix fill_uint8 = matrix(fill.dims, dtype::UInt8);
        memcpy(fill_uint8.shape(), fill.shape(), sizeof(size_m) * fill.dims);
        memcpy(fill_uint8.strides(), fill.strides(), sizeof(size_m) * fill.dims);
        fill_uint8.total_size = fill.total_size;
        fill_uint8.type = dtype::UInt8;
        fill_uint8.buffer = malloc(fill_uint8.total_size);
        matrix::copyCPUinplaceTypeCasted(fill_uint8, fill, 0);
        
        for (int y = minY; y <= maxY; ++y) {
            float dy = static_cast<float>(y - cy);
            float dy_sq = dy * dy;
            
            if (dy_sq > ry_sq) continue;
            
            // Ellipse horizontal bounds derived from x^2/a^2 + y^2/b^2 = 1
            float dx = std::sqrt(rx_sq * (1.0f - (dy_sq / ry_sq)));
            
            int minX = std::max(0, static_cast<int>(std::ceil(cx - dx)));
            int maxX = std::min(w - 1, static_cast<int>(std::floor(cx + dx)));
            
            if (minX <= maxX) {
                canvas.slice({R(y, y + 1), R(minX, maxX + 1)}) = fill_uint8;
            }
        }
        free(fill_uint8.buffer);
    }
    
    // Stroked Ellipse Scanline logic mapping
    static void drawStrokedEllipse(matrix& canvas, int cx, int cy, float rx, float ry, float stroke_width, const matrix& fill) {
        if (canvas.dims < 2 || rx <= 0.0f || ry <= 0.0f) return;
        
        int w = canvas.shape()[1];
        int h = canvas.shape()[0];
        
        float outer_rx = rx;
        float outer_ry = ry;
        float inner_rx = std::max(0.0f, rx - stroke_width);
        float inner_ry = std::max(0.0f, ry - stroke_width);
        
        int minY = std::max(0, static_cast<int>(std::ceil(cy - outer_ry)));
        int maxY = std::min(h - 1, static_cast<int>(std::floor(cy + outer_ry)));
        
        float outer_rx_sq = outer_rx * outer_rx;
        float outer_ry_sq = outer_ry * outer_ry;
        float inner_rx_sq = inner_rx * inner_rx;
        float inner_ry_sq = inner_ry * inner_ry;
        
        matrix fill_uint8 = matrix(fill.dims, dtype::UInt8);
        memcpy(fill_uint8.shape(), fill.shape(), sizeof(size_m) * fill.dims);
        memcpy(fill_uint8.strides(), fill.strides(), sizeof(size_m) * fill.dims);
        fill_uint8.total_size = fill.total_size;
        fill_uint8.type = dtype::UInt8;
        fill_uint8.buffer = malloc(fill_uint8.total_size);
        matrix::copyCPUinplaceTypeCasted(fill_uint8, fill, 0);
        
        for (int y = minY; y <= maxY; ++y) {
            float dy = static_cast<float>(y - cy);
            float dy_sq = dy * dy;
            
            if (dy_sq > outer_ry_sq) continue;
            
            float dx_outer = std::sqrt(outer_rx_sq * (1.0f - (dy_sq / outer_ry_sq)));
            int outer_minX = std::max(0, static_cast<int>(std::ceil(cx - dx_outer)));
            int outer_maxX = std::min(w - 1, static_cast<int>(std::floor(cx + dx_outer)));
            
            if (outer_minX > outer_maxX) continue;
            
            if (inner_ry_sq <= 0.0f || dy_sq >= inner_ry_sq) {
                canvas.slice({R(y, y + 1), R(outer_minX, outer_maxX + 1)}) = fill_uint8;
            } else {
                float dx_inner = std::sqrt(inner_rx_sq * (1.0f - (dy_sq / inner_ry_sq)));
                int inner_minX = std::max(0, static_cast<int>(std::ceil(cx - dx_inner)));
                int inner_maxX = std::min(w - 1, static_cast<int>(std::floor(cx + dx_inner)));
                
                if (outer_minX < inner_minX) {
                    canvas.slice({R(y, y + 1), R(outer_minX, inner_minX)}) = fill_uint8;
                }
                
                int right_start = std::max(inner_maxX + 1, outer_minX);
                if (right_start <= outer_maxX) {
                    canvas.slice({R(y, y + 1), R(right_start, outer_maxX + 1)}) = fill_uint8;
                }
            }
        }
        free(fill_uint8.buffer);
    }
    
    // --- PATH 2D SUPPORT ---
    
    static void drawPathStroked(matrix& canvas, const Path2D& path, float thickness, const matrix& fill, int segments = 50) {
        if (!path.finalized || canvas.dims < 2 || path.points.type != dtype::Float) return;
        
        float* data = static_cast<float*>(path.points.buffer);
        int pt_idx = 0;
        
        float current_x = 0.0f;
        float current_y = 0.0f;
        float start_x = 0.0f;
        float start_y = 0.0f;
        
        for (PathVerb verb : path.verbs) {
            switch(verb) {
                case PathVerb::MoveTo:
                    current_x = data[pt_idx * 2];
                    current_y = data[pt_idx * 2 + 1];
                    start_x = current_x;
                    start_y = current_y;
                    pt_idx += 1;
                    break;
                    
                case PathVerb::LineTo: {
                    float next_x = data[pt_idx * 2];
                    float next_y = data[pt_idx * 2 + 1];
                    // Our native normalized drawLine maps automatically to bounds and handles AA internally
                    drawLine(canvas, current_x, current_y, next_x, next_y, thickness, fill);
                    current_x = next_x;
                    current_y = next_y;
                    pt_idx += 1;
                    break;
                }
                    
                case PathVerb::QuadTo: {
                    float cx = data[pt_idx * 2];
                    float cy = data[pt_idx * 2 + 1];
                    float next_x = data[(pt_idx + 1) * 2];
                    float next_y = data[(pt_idx + 1) * 2 + 1];
                    drawQuadraticBezier(canvas, current_x, current_y, cx, cy, next_x, next_y, thickness, fill, segments);
                    current_x = next_x;
                    current_y = next_y;
                    pt_idx += 2;
                    break;
                }
                    
                case PathVerb::CubicTo: {
                    float cx1 = data[pt_idx * 2];
                    float cy1 = data[pt_idx * 2 + 1];
                    float cx2 = data[(pt_idx + 1) * 2];
                    float cy2 = data[(pt_idx + 1) * 2 + 1];
                    float next_x = data[(pt_idx + 2) * 2];
                    float next_y = data[(pt_idx + 2) * 2 + 1];
                    drawCubicBezier(canvas, current_x, current_y, cx1, cy1, cx2, cy2, next_x, next_y, thickness, fill, segments);
                    current_x = next_x;
                    current_y = next_y;
                    pt_idx += 3;
                    break;
                }
                    
                case PathVerb::Close:
                    drawLine(canvas, current_x, current_y, start_x, start_y, thickness, fill);
                    current_x = start_x;
                    current_y = start_y;
                    break;
            }
        }
    }
    
    struct PathEdge {
        int x0, y0, x1, y1;
    };
    
    static void drawPathFilled(matrix& canvas, const Path2D& path, const matrix& fill, int segments = 50) {
        if (!path.finalized || canvas.dims < 2 || path.points.type != dtype::Float) return;
        
        int w = canvas.shape()[1];
        int h = canvas.shape()[0];
        
        float* data = static_cast<float*>(path.points.buffer);
        int pt_idx = 0;
        
        float current_x = 0.0f, current_y = 0.0f;
        float start_x = 0.0f, start_y = 0.0f;
        
        std::vector<PathEdge> edges;
        
        auto push_edge = [&](float nx0, float ny0, float nx1, float ny1) {
            int ix0 = static_cast<int>(std::round((nx0 + 1.0f) * 0.5f * (w - 1)));
            int iy0 = static_cast<int>(std::round((1.0f - ny0) * 0.5f * (h - 1)));
            int ix1 = static_cast<int>(std::round((nx1 + 1.0f) * 0.5f * (w - 1)));
            int iy1 = static_cast<int>(std::round((1.0f - ny1) * 0.5f * (h - 1)));
            edges.push_back({ix0, iy0, ix1, iy1});
        };
        
        for (PathVerb verb : path.verbs) {
            switch(verb) {
                case PathVerb::MoveTo:
                    current_x = data[pt_idx * 2];
                    current_y = data[pt_idx * 2 + 1];
                    start_x = current_x;
                    start_y = current_y;
                    pt_idx += 1;
                    break;
                    
                case PathVerb::LineTo: {
                    float next_x = data[pt_idx * 2];
                    float next_y = data[pt_idx * 2 + 1];
                    push_edge(current_x, current_y, next_x, next_y);
                    current_x = next_x;
                    current_y = next_y;
                    pt_idx += 1;
                    break;
                }
                    
                case PathVerb::QuadTo: {
                    float cx = data[pt_idx * 2];
                    float cy = data[pt_idx * 2 + 1];
                    float next_x = data[(pt_idx + 1) * 2];
                    float next_y = data[(pt_idx + 1) * 2 + 1];
                    
                    float step = 1.0f / segments;
                    float prev_x = current_x, prev_y = current_y;
                    for (int i = 1; i <= segments; ++i) {
                        float t = i * step;
                        float inv_t = 1.0f - t;
                        float curr_nx = inv_t * inv_t * current_x + 2.0f * inv_t * t * cx + t * t * next_x;
                        float curr_ny = inv_t * inv_t * current_y + 2.0f * inv_t * t * cy + t * t * next_y;
                        push_edge(prev_x, prev_y, curr_nx, curr_ny);
                        prev_x = curr_nx;
                        prev_y = curr_ny;
                    }
                    
                    current_x = next_x;
                    current_y = next_y;
                    pt_idx += 2;
                    break;
                }
                    
                case PathVerb::CubicTo: {
                    float cx1 = data[pt_idx * 2];
                    float cy1 = data[pt_idx * 2 + 1];
                    float cx2 = data[(pt_idx + 1) * 2];
                    float cy2 = data[(pt_idx + 1) * 2 + 1];
                    float next_x = data[(pt_idx + 2) * 2];
                    float next_y = data[(pt_idx + 2) * 2 + 1];
                    
                    float step = 1.0f / segments;
                    float prev_x = current_x, prev_y = current_y;
                    for (int i = 1; i <= segments; ++i) {
                        float t = i * step;
                        float inv_t = 1.0f - t;
                        float inv_t_sq = inv_t * inv_t;
                        float t_sq = t * t;
                        float curr_nx = inv_t_sq * inv_t * current_x + 3.0f * inv_t_sq * t * cx1 + 3.0f * inv_t * t_sq * cx2 + t_sq * t * next_x;
                        float curr_ny = inv_t_sq * inv_t * current_y + 3.0f * inv_t_sq * t * cy1 + 3.0f * inv_t * t_sq * cy2 + t_sq * t * next_y;
                        push_edge(prev_x, prev_y, curr_nx, curr_ny);
                        prev_x = curr_nx;
                        prev_y = curr_ny;
                    }
                    
                    current_x = next_x;
                    current_y = next_y;
                    pt_idx += 3;
                    break;
                }
                    
                case PathVerb::Close:
                    push_edge(current_x, current_y, start_x, start_y);
                    current_x = start_x;
                    current_y = start_y;
                    break;
            }
        }
        
        // Execute universal scanline casting over all sub-contours simultaneously to evaluate winding cutouts gracefully!
        int minY = h, maxY = 0;
        for (const auto& edge : edges) {
            minY = std::min({minY, edge.y0, edge.y1});
            maxY = std::max({maxY, edge.y0, edge.y1});
        }
        minY = std::max(0, minY);
        maxY = std::min(h - 1, maxY);
        
        if (minY > maxY) return;
        
        matrix fill_uint8 = matrix(fill.dims, dtype::UInt8);
        memcpy(fill_uint8.shape(), fill.shape(), sizeof(size_m) * fill.dims);
        memcpy(fill_uint8.strides(), fill.strides(), sizeof(size_m) * fill.dims);
        fill_uint8.total_size = fill.total_size;
        fill_uint8.type = dtype::UInt8;
        fill_uint8.buffer = malloc(fill_uint8.total_size);
        matrix::copyCPUinplaceTypeCasted(fill_uint8, fill, 0);

        for (int y = minY; y <= maxY; ++y) {
            std::vector<int> nodeX;
            for (const auto& edge : edges) {
                int yi = edge.y0;
                int yj = edge.y1;
                if ((yi < y && yj >= y) || (yj < y && yi >= y)) {
                    int xi = edge.x0;
                    int xj = edge.x1;
                    nodeX.push_back(xi + static_cast<int>(static_cast<float>(y - yi) / (yj - yi) * (xj - xi)));
                }
            }
            std::sort(nodeX.begin(), nodeX.end());
            
            for (size_t i = 0; i + 1 < nodeX.size(); i += 2) {
                int startX = std::max(0, nodeX[i]);
                int endX = std::min(w - 1, nodeX[i+1]);
                if (startX <= endX) {
                    canvas.slice({R(y, y + 1), R(startX, endX + 1)}) = fill_uint8;
                }
            }
        }
        
    }
    
    // --- TEXT RENDERING ---
    
    // Callback for CGPathApply — walks each CGPath element into a Path2D
    static void _pathApplyCallback(void* info, const CGPathElement* element) {
        Path2D* path = static_cast<Path2D*>(info);
        CGPoint* p = element->points;
        
        switch (element->type) {
            case kCGPathElementMoveToPoint:
                path->moveTo((float)p[0].x, (float)p[0].y);
                break;
            case kCGPathElementAddLineToPoint:
                path->lineTo((float)p[0].x, (float)p[0].y);
                break;
            case kCGPathElementAddQuadCurveToPoint:
                path->quadTo((float)p[0].x, (float)p[0].y, (float)p[1].x, (float)p[1].y);
                break;
            case kCGPathElementAddCurveToPoint:
                path->cubicTo((float)p[0].x, (float)p[0].y, (float)p[1].x, (float)p[1].y, (float)p[2].x, (float)p[2].y);
                break;
            case kCGPathElementCloseSubpath:
                path->close();
                break;
        }
    }
    
    // Per-glyph outline cache: avoids re-extracting from CoreText every frame
    struct GlyphCacheEntry {
        std::vector<PathVerb> verbs;
        std::vector<float> points; // pairs of (x, y)
        float advance;
    };
    struct GlyphCacheKey {
        CGGlyph glyphID;
        uint32_t fontSizeX100; // fontSize * 100 as int for hashing
        bool operator==(const GlyphCacheKey& o) const { return glyphID == o.glyphID && fontSizeX100 == o.fontSizeX100; }
    };
    struct GlyphCacheHash {
        size_t operator()(const GlyphCacheKey& k) const { return std::hash<uint64_t>()((uint64_t)k.glyphID << 32 | k.fontSizeX100); }
    };
    static inline std::unordered_map<GlyphCacheKey, GlyphCacheEntry, GlyphCacheHash> _glyphCache;
    
    // Extract text glyphs into a single composed Path2D (pixel coordinates) — with caching
    static Path2D textToPath(const char* text, const char* fontName, float fontSize) {
        Path2D path;
        
        CFStringRef cfFontName = CFStringCreateWithCString(kCFAllocatorDefault, fontName, kCFStringEncodingUTF8);
        CTFontRef font = CTFontCreateWithName(cfFontName, fontSize, NULL);
        CFRelease(cfFontName);
        
        CFStringRef cfText = CFStringCreateWithCString(kCFAllocatorDefault, text, kCFStringEncodingUTF8);
        CFIndex len = CFStringGetLength(cfText);
        
        std::vector<UniChar> chars(len);
        CFStringGetCharacters(cfText, CFRangeMake(0, len), chars.data());
        CFRelease(cfText);
        
        std::vector<CGGlyph> glyphs(len);
        CTFontGetGlyphsForCharacters(font, chars.data(), glyphs.data(), len);
        
        std::vector<CGSize> advances(len);
        CTFontGetAdvancesForGlyphs(font, kCTFontOrientationHorizontal, glyphs.data(), advances.data(), len);
        
        uint32_t fsKey = (uint32_t)(fontSize * 100);
        float cursor_x = 0.0f;
        
        for (CFIndex i = 0; i < len; ++i) {
            GlyphCacheKey key = {glyphs[i], fsKey};
            auto it = _glyphCache.find(key);
            
            if (it != _glyphCache.end()) {
                // Cache hit — copy cached verbs + offset points
                const GlyphCacheEntry& cached = it->second;
                for (PathVerb v : cached.verbs) {
                    path.verbs.push_back(v);
                }
                for (size_t j = 0; j < cached.points.size(); j += 2) {
                    path._temp_points.push_back(cached.points[j] + cursor_x);
                    path._temp_points.push_back(cached.points[j + 1]);
                }
                cursor_x += cached.advance;
            } else {
                // Cache miss — extract from CoreText and store
                GlyphCacheEntry entry;
                entry.advance = advances[i].width;
                
                CGPathRef glyphPath = CTFontCreatePathForGlyph(font, glyphs[i], NULL);
                if (glyphPath) {
                    Path2D glyphP;
                    CGPathApply(glyphPath, &glyphP, _pathApplyCallback);
                    CGPathRelease(glyphPath);
                    
                    entry.verbs = glyphP.verbs;
                    // Store with Y already flipped
                    for (size_t j = 0; j < glyphP._temp_points.size(); j += 2) {
                        entry.points.push_back(glyphP._temp_points[j]);
                        entry.points.push_back(-glyphP._temp_points[j + 1]);
                    }
                    
                    for (PathVerb v : entry.verbs) {
                        path.verbs.push_back(v);
                    }
                    for (size_t j = 0; j < entry.points.size(); j += 2) {
                        path._temp_points.push_back(entry.points[j] + cursor_x);
                        path._temp_points.push_back(entry.points[j + 1]);
                    }
                }
                _glyphCache[key] = std::move(entry);
                cursor_x += advances[i].width;
            }
        }
        
        CFRelease(font);
        path.freeze();
        return path;
    }
    static void drawTextCG(matrix& canvas, const char* text, const char* fontName, float fontSize, int x, int y, const matrix& fill, simd_float2 anchor = simd_make_float2(-1.0f, 1.0f)) {
        if (canvas.dims < 2) return;

        int canvasW = canvas.shape()[1];
        int canvasH = canvas.shape()[0];
        int channels = (canvas.dims >= 3) ? canvas.shape()[2] : 1;

        // Build fill color
        matrix fill_uint8 = matrix(fill.dims, dtype::UInt8);
        memcpy(fill_uint8.shape(),   fill.shape(),   sizeof(size_m) * fill.dims);
        memcpy(fill_uint8.strides(), fill.strides(), sizeof(size_m) * fill.dims);
        fill_uint8.total_size = fill.total_size;
        fill_uint8.buffer     = malloc(fill_uint8.total_size);
        matrix::copyCPUinplaceTypeCasted(fill_uint8, fill, 0);
        uint8_t* fb = static_cast<uint8_t*>(fill_uint8.buffer);

        CGFloat r = (channels > 0) ? fb[0] / 255.0f : 0.0f;
        CGFloat g = (channels > 1) ? fb[1] / 255.0f : 0.0f;
        CGFloat b = (channels > 2) ? fb[2] / 255.0f : 0.0f;
        CGFloat a = (channels > 3) ? fb[3] / 255.0f : 1.0f;

        // Build font and measure text
        CFStringRef fontNameRef = CFStringCreateWithCString(nullptr, fontName, kCFStringEncodingUTF8);
        CTFontRef   font        = CTFontCreateWithName(fontNameRef, fontSize, nullptr);
        CFRelease(fontNameRef);

        CGColorSpaceRef cs    = CGColorSpaceCreateDeviceRGB();
        CGFloat         comps[4] = { r, g, b, a };
        CGColorRef      color = CGColorCreate(cs, comps);

        CFStringRef textRef = CFStringCreateWithCString(nullptr, text, kCFStringEncodingUTF8);

        CFStringRef       keys[2]   = { kCTFontAttributeName, kCTForegroundColorAttributeName };
        CFTypeRef         vals[2]   = { font, color };
        CFDictionaryRef   attrs     = CFDictionaryCreate(nullptr,
                                          (const void**)keys, (const void**)vals, 2,
                                          &kCFTypeDictionaryKeyCallBacks,
                                          &kCFTypeDictionaryValueCallBacks);
        CFAttributedStringRef attrStr = CFAttributedStringCreate(nullptr, textRef, attrs);
        CTLineRef line = CTLineCreateWithAttributedString(attrStr);

        // Measure typographic bounds for anchor
        CGFloat ascent, descent, leading;
        double lineWidth = CTLineGetTypographicBounds(line, &ascent, &descent, &leading);
        CGFloat lineHeight = ascent + descent; // descent is already positive from CT

        // anchor.x: -1=left, 0=center, 1=right
        // anchor.y:  1=top,  0=center, -1=bottom
        float anchorNormX = (anchor.x + 1.0f) * 0.5f;
        float anchorNormY = (1.0f - anchor.y) * 0.5f;  // 0=top, 1=bottom

        // Compute text origin in canvas Y-down coordinates
        // anchorNormX: 0=left, 0.5=center, 1=right
        // anchorNormY: 0=top, 0.5=center, 1=bottom
        float textOriginX = x - anchorNormX * lineWidth;
        // In canvas Y-down space, the text top is at y when anchorNormY=0
        // The baseline sits at (top + ascent) in Y-down coords
        // top = y - anchorNormY * lineHeight  (so top shifts up for center/bottom anchors)
        float textTop = y - anchorNormY * lineHeight;
        // In the flipped CG context, Y-down y maps directly, and baseline = top + ascent
        float textOriginY = textTop + ascent;

        // Create bitmap context directly over canvas buffer
        // Assumes canvas is HWC uint8, contiguous, RGB or RGBA
        CGBitmapInfo bitmapInfo = (channels == 4)
            ? kCGImageAlphaPremultipliedLast
            : kCGImageAlphaNoneSkipLast;

        // CG needs 4 channels minimum for RGB; if canvas is 3-channel we need a scratch buffer
        bool needScratch = (channels != 4);
        uint8_t* cgBuf   = nullptr;
        size_t   cgStride = canvasW * 4;

        if (needScratch) {
            cgBuf = static_cast<uint8_t*>(calloc(canvasH * cgStride, 1));
            // Copy canvas into scratch, padding alpha=255
            uint8_t* src = static_cast<uint8_t*>(canvas.buffer);
            for (int row = 0; row < canvasH; ++row) {
                for (int col = 0; col < canvasW; ++col) {
                    int si = row * canvas.strides()[0] + col * canvas.strides()[1];
                    int di = row * cgStride + col * 4;
                    for (int c = 0; c < std::min(channels, 3); ++c)
                        cgBuf[di + c] = src[si + (canvas.dims >= 3 ? c * canvas.strides()[2] : 0)];
                    cgBuf[di + 3] = 255;
                }
            }
        } else {
            // Use canvas buffer directly (must be contiguous row-major RGBA)
            cgBuf = static_cast<uint8_t*>(canvas.buffer);
            cgStride = canvas.strides()[0]; // bytes per row
        }

        CGContextRef ctx = CGBitmapContextCreate(
            cgBuf, canvasW, canvasH,
            8, cgStride,
            cs, kCGImageAlphaPremultipliedLast
        );

        if (!ctx) {
            fprintf(stderr, "drawText: failed to create CGContext\n");
            if (needScratch) free(cgBuf);
            goto cleanup;
        }

        // Flip CG from Y-up to Y-down so canvas pixel coords map directly
        CGContextSaveGState(ctx);
        CGContextTranslateCTM(ctx, 0, canvasH);
        CGContextScaleCTM(ctx, 1.0, -1.0);

        // CoreText needs the text matrix set to identity (it defaults to Y-up),
        // but since we flipped the entire context, we need to flip the text matrix
        // too so glyphs render right-side-up.
        CGContextSetTextMatrix(ctx, CGAffineTransformMake(1.0, 0.0, 0.0, -1.0, 0.0, 0.0));
        CGContextSetTextPosition(ctx, textOriginX, textOriginY);
        CTLineDraw(line, ctx);
        CGContextRestoreGState(ctx);

        // Copy scratch back to canvas if needed
        if (needScratch) {
            uint8_t* dst = static_cast<uint8_t*>(canvas.buffer);
            for (int row = 0; row < canvasH; ++row) {
                for (int col = 0; col < canvasW; ++col) {
                    int di = row * canvas.strides()[0] + col * canvas.strides()[1];
                    int si = row * cgStride + col * 4;
                    for (int c = 0; c < channels; ++c)
                        dst[di + (canvas.dims >= 3 ? c * canvas.strides()[2] : 0)] = cgBuf[si + c];
                }
            }
            free(cgBuf);
        }

        CGContextRelease(ctx);

    cleanup:
//        CTLineRelease(line);
        CFRelease(attrStr);
        CFRelease(attrs);
        CFRelease(textRef);
        CFRelease(color);
        CGColorSpaceRelease(cs);
        CFRelease(font);
    }
    // Draw text with 4x supersampled anti-aliasing
    // anchor: simd_float2 in [-1,1] range — which point of the text bbox pins to (x,y)
    //   (-1, 1) = top-left (default),  (0, 0) = center,  (1, -1) = bottom-right
    static void drawText(matrix& canvas, const char* text, const char* fontName, float fontSize, int x, int y, const matrix& fill, simd_float2 anchor = simd_make_float2(-1.0f, 1.0f), int ssaa = 4) {
        if (canvas.dims < 2) return;
        
        int channels = (canvas.dims >= 3) ? canvas.shape()[2] : 1;
        int segments = 80;
        
        // Extract glyphs at supersampled resolution
        Path2D path = textToPath(text, fontName, fontSize * ssaa);
        if (!path.finalized || path.points.shape()[0] < 2) return;
        
        float* pts = static_cast<float*>(path.points.buffer);
        int numPts = path.points.shape()[0];
        
        // Compute raw bounding box of the path (before any offset)
        float rawMinX = pts[0], rawMaxX = pts[0], rawMinY = pts[1], rawMaxY = pts[1];
        for (int i = 1; i < numPts; ++i) {
            rawMinX = std::min(rawMinX, pts[i*2]);
            rawMaxX = std::max(rawMaxX, pts[i*2]);
            rawMinY = std::min(rawMinY, pts[i*2+1]);
            rawMaxY = std::max(rawMaxY, pts[i*2+1]);
        }
        float textW = rawMaxX - rawMinX;
        float textH = rawMaxY - rawMinY;
        
        // Anchor offset: map [-1,1] anchor to [0,1] normalized position in bbox
        // anchor.x: -1=left, 0=center, 1=right
        // anchor.y:  1=top, 0=center, -1=bottom
        float anchorNormX = (anchor.x + 1.0f) * 0.5f; // 0=left, 0.5=center, 1=right
        float anchorNormY = (1.0f - anchor.y) * 0.5f;  // 0=top, 0.5=center, 1=bottom
        
        // The anchor point within the bbox that should sit at (x, y)
        float anchorOffsetX = rawMinX + anchorNormX * textW;
        float anchorOffsetY = rawMinY + anchorNormY * textH;
        
        // Offset all points so the anchor maps to (x*ssaa, y*ssaa)
        float dx = static_cast<float>(x * ssaa) - anchorOffsetX;
        float dy = static_cast<float>(y * ssaa) - anchorOffsetY;
        for (int i = 0; i < numPts; ++i) {
            pts[i * 2]     += dx;
            pts[i * 2 + 1] += dy;
        }
        
        // Compute bounding box of path in hi-res space
        float fminX = pts[0], fmaxX = pts[0], fminY = pts[1], fmaxY = pts[1];
        for (int i = 1; i < numPts; ++i) {
            fminX = std::min(fminX, pts[i*2]);
            fmaxX = std::max(fmaxX, pts[i*2]);
            fminY = std::min(fminY, pts[i*2+1]);
            fmaxY = std::max(fmaxY, pts[i*2+1]);
        }
        
        int hiMinX = std::max(0, (int)std::floor(fminX) - 2);
        int hiMinY = std::max(0, (int)std::floor(fminY) - 2);
        int hiMaxX = (int)std::ceil(fmaxX) + 2;
        int hiMaxY = (int)std::ceil(fmaxY) + 2;
        int hiW = hiMaxX - hiMinX + 1;
        int hiH = hiMaxY - hiMinY + 1;
        if (hiW <= 0 || hiH <= 0) return;
        
        // Allocate hi-res 1-channel coverage buffer
        uint8_t* hiBuf = static_cast<uint8_t*>(calloc(hiW * hiH, 1));
        
        // Flatten path into edges in hi-res coords
        int pt_idx = 0;
        float cur_x = 0, cur_y = 0, st_x = 0, st_y = 0;
        struct HiEdge { int x0, y0, x1, y1, mn, mx; };
        std::vector<HiEdge> edges;
        edges.reserve(path.verbs.size() * segments);
        
        auto pe = [&](float a, float b, float c, float d) {
            int iy0 = (int)(b + 0.5f), iy1 = (int)(d + 0.5f);
            if (iy0 == iy1) return;
            int ix0 = (int)(a + 0.5f), ix1 = (int)(c + 0.5f);
            edges.push_back({ix0, iy0, ix1, iy1, iy0<iy1?iy0:iy1, iy0>iy1?iy0:iy1});
        };
        
        for (PathVerb verb : path.verbs) {
            switch(verb) {
                case PathVerb::MoveTo:
                    cur_x = pts[pt_idx*2]; cur_y = pts[pt_idx*2+1];
                    st_x = cur_x; st_y = cur_y; pt_idx++; break;
                case PathVerb::LineTo: {
                    float nx = pts[pt_idx*2], ny = pts[pt_idx*2+1];
                    pe(cur_x, cur_y, nx, ny);
                    cur_x = nx; cur_y = ny; pt_idx++; break;
                }
                case PathVerb::QuadTo: {
                    float cx = pts[pt_idx*2], cy = pts[pt_idx*2+1];
                    float nx = pts[(pt_idx+1)*2], ny = pts[(pt_idx+1)*2+1];
                    float px = cur_x, py = cur_y;
                    for (int s = 1; s <= segments; s++) {
                        float t = (float)s/segments, inv = 1-t;
                        float qx = inv*inv*cur_x + 2*inv*t*cx + t*t*nx;
                        float qy = inv*inv*cur_y + 2*inv*t*cy + t*t*ny;
                        pe(px, py, qx, qy); px = qx; py = qy;
                    }
                    cur_x = nx; cur_y = ny; pt_idx += 2; break;
                }
                case PathVerb::CubicTo: {
                    float c1x = pts[pt_idx*2], c1y = pts[pt_idx*2+1];
                    float c2x = pts[(pt_idx+1)*2], c2y = pts[(pt_idx+1)*2+1];
                    float nx = pts[(pt_idx+2)*2], ny = pts[(pt_idx+2)*2+1];
                    float px = cur_x, py = cur_y;
                    for (int s = 1; s <= segments; s++) {
                        float t = (float)s/segments, inv = 1-t;
                        float inv2 = inv*inv, t2 = t*t;
                        float qx = inv2*inv*cur_x + 3*inv2*t*c1x + 3*inv*t2*c2x + t2*t*nx;
                        float qy = inv2*inv*cur_y + 3*inv2*t*c1y + 3*inv*t2*c2y + t2*t*ny;
                        pe(px, py, qx, qy); px = qx; py = qy;
                    }
                    cur_x = nx; cur_y = ny; pt_idx += 3; break;
                }
                case PathVerb::Close:
                    pe(cur_x, cur_y, st_x, st_y);
                    cur_x = st_x; cur_y = st_y; break;
            }
        }
        
        if (edges.empty()) { free(hiBuf); return; }
        
        // Sort edges by yMin for sweep-line
        std::sort(edges.begin(), edges.end(), [](const HiEdge& a, const HiEdge& b) { return a.mn < b.mn; });
        
        int eMinY = std::max(0, edges.front().mn - hiMinY);
        int eMaxY = 0;
        for (const auto& e : edges) { int ey = e.mx - hiMinY; if (ey > eMaxY) eMaxY = ey; }
        if (eMaxY >= hiH) eMaxY = hiH - 1;
        
        int edgeStart = 0;
        int nodeXBuf[128];
        
        for (int row = eMinY; row <= eMaxY; ++row) {
            int absY = row + hiMinY;
            int nc = 0;
            for (int ei = edgeStart; ei < (int)edges.size(); ++ei) {
                const HiEdge& e = edges[ei];
                if (e.mn - hiMinY > row) break;
                if (e.mx - hiMinY < row) { if (ei == edgeStart) edgeStart++; continue; }
                int yi = e.y0, yj = e.y1;
                if ((yi < absY && yj >= absY) || (yj < absY && yi >= absY)) {
                    if (nc < 128) nodeXBuf[nc++] = e.x0 + (int)((float)(absY - yi) / (yj - yi) * (e.x1 - e.x0));
                }
            }
            for (int i = 1; i < nc; ++i) { int k = nodeXBuf[i], j = i-1; while (j>=0 && nodeXBuf[j]>k) { nodeXBuf[j+1]=nodeXBuf[j]; j--; } nodeXBuf[j+1]=k; }
            for (int i = 0; i+1 < nc; i += 2) {
                int sX = nodeXBuf[i] - hiMinX; if (sX < 0) sX = 0;
                int eX = nodeXBuf[i+1] - hiMinX; if (eX >= hiW) eX = hiW - 1;
                if (sX <= eX) memset(hiBuf + row * hiW + sX, 255, eX - sX + 1);
            }
        }
        
        // Downsample and alpha-blend onto canvas
        int canvasW = canvas.shape()[1];
        int canvasH = canvas.shape()[0];
        uint8_t* canvas_data = static_cast<uint8_t*>(canvas.buffer);
        
        uint8_t fill_bytes[32] = {};
        matrix fill_uint8 = matrix(fill.dims, dtype::UInt8);
        memcpy(fill_uint8.shape(), fill.shape(), sizeof(size_m) * fill.dims);
        memcpy(fill_uint8.strides(), fill.strides(), sizeof(size_m) * fill.dims);
        fill_uint8.total_size = fill.total_size;
        fill_uint8.type = dtype::UInt8;
        fill_uint8.buffer = malloc(fill_uint8.total_size);
        matrix::copyCPUinplaceTypeCasted(fill_uint8, fill, 0);
        memcpy(fill_bytes, fill_uint8.buffer, sizeof(uint8_t) * fill_uint8.total_size);
        
        float invSamples = 1.0f / (ssaa * ssaa);
        int outMinX = std::max(0, hiMinX / ssaa);
        int outMinY = std::max(0, hiMinY / ssaa);
        int outMaxX = std::min(canvasW - 1, (hiMaxX + ssaa - 1) / ssaa);
        int outMaxY = std::min(canvasH - 1, (hiMaxY + ssaa - 1) / ssaa);
        
        for (int oy = outMinY; oy <= outMaxY; ++oy) {
            for (int ox = outMinX; ox <= outMaxX; ++ox) {
                int sum = 0;
                for (int sy = 0; sy < ssaa; ++sy) {
                    int hiY = oy * ssaa + sy - hiMinY;
                    if (hiY < 0 || hiY >= hiH) continue;
                    for (int sx = 0; sx < ssaa; ++sx) {
                        int hiX = ox * ssaa + sx - hiMinX;
                        if (hiX < 0 || hiX >= hiW) continue;
                        sum += hiBuf[hiY * hiW + hiX];
                    }
                }
                if (sum == 0) continue;
                float alpha = (sum * invSamples) / 255.0f;
                int pidx = oy * canvas.strides()[0] + ox * canvas.strides()[1];
                if (alpha >= 0.99f) {
                    for (int c = 0; c < channels; ++c) {
                        int co = (canvas.dims >= 3) ? c * canvas.strides()[2] : 0;
                        canvas_data[pidx + co] = fill_bytes[c];
                    }
                } else {
                    for (int c = 0; c < channels; ++c) {
                        int co = (canvas.dims >= 3) ? c * canvas.strides()[2] : 0;
                        canvas_data[pidx + co] = (uint8_t)(canvas_data[pidx + co] * (1.0f - alpha) + fill_bytes[c] * alpha);
                    }
                }
            }
        }
        free(hiBuf);
    }
    
    // --- PLOT RENDERING ---
    
    static void drawPlot(matrix& canvas, const Plot& plot) {
        if (canvas.dims < 2) return;
        
        int canvasW = canvas.shape()[1];
        int canvasH = canvas.shape()[0];
        
        // 1. Fill entire background
        drawRect(canvas, 0, 0, canvasW, canvasH, plot.bgColor);
        
        // Plot area bounds (pixel space)
        int plotX0 = plot.marginLeft;
        int plotY0 = plot.marginTop;
        int plotX1 = canvasW - plot.marginRight;
        int plotY1 = canvasH - plot.marginBottom;
        int plotW = plotX1 - plotX0;
        int plotH = plotY1 - plotY0;
        
        if (plotW <= 0 || plotH <= 0) return;
        
        // 2. Fill plot area background
        drawRect(canvas, plotX0, plotY0, plotX1, plotY1, plot.plotAreaColor);
        
        // Data → pixel coordinate mapping lambdas
        float xRange = plot.xAxis.max - plot.xAxis.min;
        float yRange = plot.yAxis.max - plot.yAxis.min;
        if (xRange == 0.0f) xRange = 1.0f;
        if (yRange == 0.0f) yRange = 1.0f;
        
        auto dataToPixelX = [&](float dataX) -> int {
            return plotX0 + static_cast<int>((dataX - plot.xAxis.min) / xRange * plotW);
        };
        auto dataToPixelY = [&](float dataY) -> int {
            return plotY1 - static_cast<int>((dataY - plot.yAxis.min) / yRange * plotH);
        };
        
        // 3. Draw grid lines
        if (plot.showGrid && plot.xAxis.tickCount > 0) {
            for (int t = 0; t <= plot.xAxis.tickCount; ++t) {
                float val = plot.xAxis.min + (float)t / plot.xAxis.tickCount * xRange;
                int px = dataToPixelX(val);
                if (px >= plotX0 && px <= plotX1) {
                    drawLine(canvas, px, plotY0, px, plotY1, plot.gridLineWidth, plot.gridColor);
                }
            }
        }
        if (plot.showGrid && plot.yAxis.tickCount > 0) {
            for (int t = 0; t <= plot.yAxis.tickCount; ++t) {
                float val = plot.yAxis.min + (float)t / plot.yAxis.tickCount * yRange;
                int py = dataToPixelY(val);
                if (py >= plotY0 && py <= plotY1) {
                    drawLine(canvas, plotX0, py, plotX1, py, plot.gridLineWidth, plot.gridColor);
                }
            }
        }
        
        // 4. Draw axis frame lines (left + bottom)
        drawLine(canvas, plotX0, plotY0, plotX0, plotY1, plot.axisLineWidth, plot.axisColor);
        drawLine(canvas, plotX0, plotY1, plotX1, plotY1, plot.axisLineWidth, plot.axisColor);
        // Optional top + right for a full box
        drawLine(canvas, plotX1, plotY0, plotX1, plotY1, 1.0f, plot.gridColor);
        drawLine(canvas, plotX0, plotY0, plotX1, plotY0, 1.0f, plot.gridColor);
        
        // 5. Draw tick labels
        char tickBuf[32];
        
        // X-axis ticks
        for (int t = 0; t <= plot.xAxis.tickCount; ++t) {
            float val = plot.xAxis.min + (float)t / plot.xAxis.tickCount * xRange;
            int px = dataToPixelX(val);
            
            // Tick mark
            drawLine(canvas, px, plotY1, px, plotY1 + 5, plot.axisLineWidth, plot.axisColor);
            
            // Label
            if (val == (int)val) snprintf(tickBuf, sizeof(tickBuf), "%d", (int)val);
            else snprintf(tickBuf, sizeof(tickBuf), "%.1f", val);
            
            drawTextCG(canvas, tickBuf, plot.xAxis.fontName, plot.xAxis.tickFontSize,
                     px, plotY1 + 8, plot.axisColor, simd_make_float2(0.0f, 1.0f));
        }
        
        // Y-axis ticks
        for (int t = 0; t <= plot.yAxis.tickCount; ++t) {
            float val = plot.yAxis.min + (float)t / plot.yAxis.tickCount * yRange;
            int py = dataToPixelY(val);
            
            // Tick mark
            drawLine(canvas, plotX0 - 5, py, plotX0, py, plot.axisLineWidth, plot.axisColor);
            
            // Label
            if (val == (int)val) snprintf(tickBuf, sizeof(tickBuf), "%d", (int)val);
            else snprintf(tickBuf, sizeof(tickBuf), "%.1f", val);
            
            drawTextCG(canvas, tickBuf, plot.yAxis.fontName, plot.yAxis.tickFontSize,
                     plotX0 - 8, py, plot.axisColor, simd_make_float2(1.0f, 0.0f));
        }
        
        // 6. Axis titles
        if (strlen(plot.xAxis.title) > 0) {
            drawTextCG(canvas, plot.xAxis.title, plot.xAxis.fontName, plot.xAxis.fontSize,
                     (plotX0 + plotX1) / 2, plotY1 + 35, plot.axisColor, simd_make_float2(0.0f, 1.0f));
        }
        if (strlen(plot.yAxis.title) > 0) {
            // Y-axis title drawn vertically is complex; place it left of ticks horizontally for now
            drawTextCG(canvas, plot.yAxis.title, plot.yAxis.fontName, plot.yAxis.fontSize,
                     5, (plotY0 + plotY1) / 2, plot.axisColor, simd_make_float2(-1.0f, 0.0f));
        }
        
        // 7. Plot title
        if (strlen(plot.title) > 0) {
            drawTextCG(canvas, plot.title, plot.fontName, plot.titleFontSize,
                     (plotX0 + plotX1) / 2, plotY0 - 10, plot.axisColor, simd_make_float2(0.0f, -1.0f));
        }
        
        // 8. Render each data series
        for (int si = 0; si < plot.seriesCount; ++si) {
            const PlotSeries& s = plot.series[si];
            
            // Get data as float pointers
            matrix fx = s.x, fy = s.y;
            if (s.x.type != dtype::Float) {
                fx = matrix::withShape({(size_m)s.x.total_size}, dtype::Float);
                matrix::copyCPUinplaceTypeCasted(fx, s.x, 0);
            }
            if (s.y.type != dtype::Float) {
                fy = matrix::withShape({(size_m)s.y.total_size}, dtype::Float);
                matrix::copyCPUinplaceTypeCasted(fy, s.y, 0);
            }
            
            float* xdata = static_cast<float*>(fx.buffer);
            float* ydata = static_cast<float*>(fy.buffer);
            int count = std::min(fx.total_size, fy.total_size);
            if (count < 1) continue;
            
            // Draw lines between consecutive data points
            if (s.style == PlotStyle::Line || s.style == PlotStyle::Both) {
                for (int i = 0; i < count - 1; ++i) {
                    float lx0 = (float)dataToPixelX(xdata[i]);
                    float ly0 = (float)dataToPixelY(ydata[i]);
                    float lx1 = (float)dataToPixelX(xdata[i + 1]);
                    float ly1 = (float)dataToPixelY(ydata[i + 1]);
                    
                    // Cohen-Sutherland line clipping to plot area
                    float cxmin = (float)plotX0, cymin = (float)plotY0;
                    float cxmax = (float)plotX1, cymax = (float)plotY1;
                    auto outcode = [&](float cx, float cy) -> int {
                        int code = 0;
                        if (cx < cxmin) code |= 1;
                        else if (cx > cxmax) code |= 2;
                        if (cy < cymin) code |= 4;
                        else if (cy > cymax) code |= 8;
                        return code;
                    };
                    int oc0 = outcode(lx0, ly0);
                    int oc1 = outcode(lx1, ly1);
                    bool accept = false;
                    for (;;) {
                        if (!(oc0 | oc1)) { accept = true; break; }
                        if (oc0 & oc1) break;
                        int oc = oc0 ? oc0 : oc1;
                        float nx, ny;
                        if (oc & 8)      { nx = lx0 + (lx1-lx0)*(cymax-ly0)/(ly1-ly0); ny = cymax; }
                        else if (oc & 4) { nx = lx0 + (lx1-lx0)*(cymin-ly0)/(ly1-ly0); ny = cymin; }
                        else if (oc & 2) { ny = ly0 + (ly1-ly0)*(cxmax-lx0)/(lx1-lx0); nx = cxmax; }
                        else             { ny = ly0 + (ly1-ly0)*(cxmin-lx0)/(lx1-lx0); nx = cxmin; }
                        if (oc == oc0) { lx0 = nx; ly0 = ny; oc0 = outcode(lx0, ly0); }
                        else           { lx1 = nx; ly1 = ny; oc1 = outcode(lx1, ly1); }
                    }
                    if (accept) {
                        drawLine(canvas, (int)lx0, (int)ly0, (int)lx1, (int)ly1, s.lineWidth, s.color);
                    }
                }
            }
            
            // Draw scatter points as filled circles
            if (s.style == PlotStyle::Scatter || s.style == PlotStyle::Both) {
                for (int i = 0; i < count; ++i) {
                    int px = dataToPixelX(xdata[i]);
                    int py = dataToPixelY(ydata[i]);
                    
                    // Clip
                    if (px >= plotX0 - (int)s.pointRadius && px <= plotX1 + (int)s.pointRadius &&
                        py >= plotY0 - (int)s.pointRadius && py <= plotY1 + (int)s.pointRadius) {
                        drawCircle(canvas, px, py, s.pointRadius, s.color);
                    }
                }
            }
        }
    }

};

#endif /* cpu_rasteriser_hpp */

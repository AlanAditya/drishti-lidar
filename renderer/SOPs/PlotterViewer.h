#pragma once
#include "Viewer.cpp"
#include "Controllers.h"
#include "matrix.h"
#include <algorithm>
#include <cmath>
#include <vector>

// Helper to interpolate between colors for a premium dark theme
struct RGB { float r, g, b; };

inline RGB get_colormap_inferno(float t) {
    // A simplified Inferno-like colormap (dark blue -> purple -> magenta -> orange -> yellow)
    t = std::max(0.0f, std::min(1.0f, t));
    
    std::vector<RGB> colors = {
        {0.05f, 0.05f, 0.15f}, // Deep dark blue
        {0.2f, 0.0f, 0.4f},    // Purple
        {0.6f, 0.1f, 0.4f},    // Magenta
        {0.9f, 0.3f, 0.1f},    // Orange-red
        {1.0f, 0.8f, 0.2f},    // Yellow
        {1.0f, 1.0f, 0.8f}     // Bright white-yellow
    };
    
    float scaled_t = t * (colors.size() - 1);
    int idx1 = std::min((int)colors.size() - 2, (int)scaled_t);
    int idx2 = idx1 + 1;
    float frac = scaled_t - idx1;
    
    return {
        colors[idx1].r + frac * (colors[idx2].r - colors[idx1].r),
        colors[idx1].g + frac * (colors[idx2].g - colors[idx1].g),
        colors[idx1].b + frac * (colors[idx2].b - colors[idx1].b)
    };
}



inline matrix get_colormap_inferno(matrix t) {
    // A simplified Inferno-like colormap (dark blue -> purple -> magenta -> orange -> yellow)
    t = t.clamp(0.0, 1.0);

    static matrix colors = []() {
        matrix c = {
            {0.05f, 0.05f, 0.15f}, // Deep dark blue
            {0.2f, 0.0f, 0.4f},    // Purple
            {0.6f, 0.1f, 0.4f},    // Magenta
            {0.9f, 0.3f, 0.1f},    // Orange-red
            {1.0f, 0.8f, 0.2f},    // Yellow
            {1.0f, 1.0f, 0.8f}     // Bright white-yellow
        };
        c.buildMetalBuffer();
        c.begin_refcount();
        return c;
    }();

    matrix scaled_t = t * (float)(colors.shape()[0] - 1);

    
    matrix idx1 = matrix::min(matrix::scalar((int)(colors.shape()[0]) - 2), scaled_t.astype(dtype::Int32));
    matrix idx2 = idx1 + 1;
    matrix frac = scaled_t - idx1.astype(dtype::Float);
    matrix p1 = colors.take(idx1, 0);
    matrix p2 = colors.take(idx2, 0);
    
    // Unsqueeze frac so it broadcasts over the RGB dimension (e.g. from HxW to HxWx1)
    frac = frac.unsqueeze(-1);
    
    return p1 + (p2 - p1) * frac;
}


inline simd_float4x4 p_translation(float x, float y, float z) {
    simd_float4x4 m = matrix_identity_float4x4;
    m.columns[3] = simd_make_float4(x, y, z, 1.0f);
    return m;
}

inline simd_float4x4 p_scale(float sx, float sy, float sz) {
    simd_float4x4 m = matrix_identity_float4x4;
    m.columns[0][0] = sx;
    m.columns[1][1] = sy;
    m.columns[2][2] = sz;
    return m;
}

class PlotterViewer : public Viewer {
public:
    PlotterViewer() {
        // Initialize the basic scene nodes
        
        // 1. The main amplitude plot (Quad)
        QuadController plot_quad("AmplitudePlot");
        // default transform is identity. Width/Height 1.0f.
        nodes.push_back(plot_quad.node);
        
        // 2. The scale bar (Quad)
        QuadController scale_bar("ScaleBar");
        
        // Position scale bar on the left: x = -0.6, width = 0.05
        simd_float4x4 s_scale = p_scale(0.05f, 1.0f, 1.0f);
        simd_float4x4 s_trans = p_translation(-0.6f, 0.0f, 0.0f);
        scale_bar.node->local_transform.SIMD_MAT(0) = matrix_multiply(s_trans, s_scale);
        
        // Generate scale bar texture (1D gradient)
        matrix scale_img(3, dtype::UInt8);
        scale_img.shape()[0] = 256;
        scale_img.shape()[1] = 16;
        scale_img.shape()[2] = 4;
        scale_img.calcStrides();
        scale_img.total_size = 256 * 16 * 4;
        scale_img.buffer = new uint8_t[scale_img.total_size];
        uint8_t* scale_buf = (uint8_t*)scale_img.buffer;
        
        for (int y = 0; y < 256; ++y) {
            float t = y / 255.0f;
            RGB c = get_colormap_inferno(t);
            uint8_t r = (uint8_t)(c.r * 255);
            uint8_t g = (uint8_t)(c.g * 255);
            uint8_t b = (uint8_t)(c.b * 255);
            
            for (int x = 0; x < 16; ++x) {
                int idx = (y * 16 + x) * 4;
                scale_buf[idx + 0] = r;
                scale_buf[idx + 1] = g;
                scale_buf[idx + 2] = b;
                scale_buf[idx + 3] = 255;
            }
        }
        scale_img.buildMetalBuffer();
        
        scale_bar.node->material.has_texture = true;
        scale_bar.node->material.texture = scale_img.ToMTLTexture();
        nodes.push_back(scale_bar.node);
        
        // 3. Rectangle frame (Boundary)
        // A slightly larger solid quad placed slightly behind the plot image (z = -0.01)
        QuadController frame_quad("PlotFrame");
        simd_float4x4 f_scale = p_scale(1.02f, 1.02f, 1.0f);
        simd_float4x4 f_trans = p_translation(0.0f, 0.0f, +0.01f);
        frame_quad.node->local_transform.SIMD_MAT(0) = matrix_multiply(f_trans, f_scale);
        
        // GeoNode creates Material with solid white color by default.
        // We just need to make sure it doesn't try to load a texture.
        frame_quad.node->material.has_texture = false; 
        
        nodes.push_back(frame_quad.node);
    }
    
    void update_plot(const matrix& amplitude_map, float fixed_min = NAN, float fixed_max = NAN) {
        // amplitude_map is expected to be [H, W] float
        if (amplitude_map.dims != 2 || amplitude_map.type != dtype::Float) return;
        
        // Force the matrix DAG to evaluate and sync any GPU computations down to the CPU buffer
        // Otherwise, reading .buffer directly will yield stale or uninitialized data!
        size_m H = amplitude_map.shape()[0];
        size_m W = amplitude_map.shape()[1];
        
        
        // Find min and max for normalization
        float min_val = std::isnan(fixed_min) ? std::numeric_limits<float>::max() : fixed_min;
        float max_val = std::isnan(fixed_max) ? std::numeric_limits<float>::lowest() : fixed_max;
        
        if (std::isnan(fixed_min) || std::isnan(fixed_max)) {
            if (amplitude_map.total_size > 1024) {
                // Now this will correctly wait for the GPU!
                if (std::isnan(fixed_min)) min_val = amplitude_map.min().at<float>();
                if (std::isnan(fixed_max)) max_val = amplitude_map.max().at<float>();
            } else {
                float* amp_buf = (float*)amplitude_map.buffer;
                for (size_t i = 0; i < amplitude_map.total_size; ++i) {
                    float val = amp_buf[i];
                    if (std::isnan(fixed_min) && val < min_val) min_val = val;
                    if (std::isnan(fixed_max) && val > max_val) max_val = val;
                }
            }
        }
        
        float range = max_val - min_val;
        if (range == 0.0f) range = 1.0f;
        // Map to texture
        matrix color_img(3, dtype::UInt8);
        if (amplitude_map.total_size > 1024) {
            matrix normalised_map = (amplitude_map - min_val) / range;
            normalised_map = get_colormap_inferno(normalised_map);
            normalised_map = matrix::concat({normalised_map, matrix::ones({normalised_map.shape()[0], normalised_map.shape()[1], 1}, dtype::Float) }, 2) * 255.0f;
            normalised_map = normalised_map.astype(dtype::UInt8);
            normalised_map.eval_metal();
            nodes[0]->material.texture = normalised_map.ToMTLTexture();
        } else {
            const_cast<matrix&>(amplitude_map).update_from_trace();
            float* amp_buf = (float*) amplitude_map.buffer;
            color_img.shape()[0] = H;
            color_img.shape()[1] = W;
            color_img.shape()[2] = 4;
            color_img.calcStrides();
            color_img.total_size = H * W * 4;
            color_img.buffer = new uint8_t[color_img.total_size];
            
            uint8_t* color_buf = (uint8_t*)color_img.buffer;
            
            for (size_t i = 0; i < amplitude_map.total_size; ++i) {
                float t = (amp_buf[i] - min_val) / range;
                RGB c = get_colormap_inferno(t);
                
                color_buf[i * 4 + 0] = (uint8_t)(c.r * 255.0f);
                color_buf[i * 4 + 1] = (uint8_t)(c.g * 255.0f);
                color_buf[i * 4 + 2] = (uint8_t)(c.b * 255.0f);
                color_buf[i * 4 + 3] = 255;
            }
            
            color_img.buildMetalBuffer();
            nodes[0]->material.texture = color_img.ToMTLTexture();
        }
        // Update the material of the first node (the plot quad)
        nodes[0]->material.has_texture = true;
    }
};

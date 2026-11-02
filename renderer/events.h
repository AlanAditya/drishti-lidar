#pragma once
#include <simd/simd.h>
#include <tuple>

// Input events belong to the renderer, not to matrix.
enum class DrishtiMouseEvent { Drag, Hover, Zoom, Scroll, Tap };
struct DrishtiViewerEvent {
    DrishtiMouseEvent type;
    simd_float2 normalized_pos = {0, 0};
    simd_float2 prev_normalized_pos = {0, 0};
    simd_float2 delta;
    float zoom_magnitude;
    simd_float3 ray_origin, ray_dir, prev_ray_origin, prev_ray_dir;
};

template <typename Cam>
inline std::tuple<simd_float3, simd_float3, simd_float2>
drishti_ray_from_screen_space(simd_float2 view_coords, simd_float2 view_bounds, Cam& cam) {
    simd_float2 pos = 2 * view_coords / view_bounds - 1;
    simd_float4x4 inv = simd_mul(cam.inverseViewMatrix, cam.inverseProjectionMatrix);
    simd_float4 a = simd_mul(inv, simd_make_float4(pos.x, pos.y, 0, 1));
    simd_float4 b = simd_mul(inv, simd_make_float4(pos.x, pos.y, 1, 1));
    simd_float3 near_pt = simd_make_float3(a.x / a.w, a.y / a.w, a.z / a.w);
    simd_float3 far_pt = simd_make_float3(b.x / b.w, b.y / b.w, b.z / b.w);
    return {near_pt, far_pt - near_pt, pos};
}

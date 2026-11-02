#pragma once
#include "../matrix/matrix.h"
#include <simd/simd.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

// Colours and gradients, manim style.
//
//   g->bar(edges, heights, col::gradient({col::BLUE, col::PURPLE}, B));     // by position, left to right
//   g->bar(edges, heights, col::map(heights, {col::BLUE_E, col::BLUE, col::PURPLE_A}));  // by height
//   node->material.colors = col::rgba(col::TEAL);
//
// Every colour here is LINEAR rgba, which is what material colours are: the renderers draw into
// BGRA8Unorm_sRGB, so the GPU treats shader output as linear light and encodes it to sRGB for the
// screen. A hex code (sRGB, as in manim, CSS or a colour picker) goes through hex() / from_srgb(),
// which decode it; written in raw, #236B8E would come out much paler. Gradients interpolate in
// OKLab (Björn Ottosson's perceptual space, defined on linear rgb), so blue -> purple stays vivid
// instead of going grey in the middle as a plain rgb lerp does.
namespace col {

inline float srgb_to_linear(float x) { return x <= 0.04045f ? x / 12.92f : std::pow((x + 0.055f) / 1.055f, 2.4f); }
inline float linear_to_srgb(float x) {
    x = std::clamp(x, 0.0f, 1.0f);
    return x <= 0.0031308f ? 12.92f * x : 1.055f * std::pow(x, 1.0f / 2.4f) - 0.055f;
}

// sRGB rgba in 0..1 (what a colour picker shows) -> linear; alpha is already linear.
inline simd_float4 from_srgb(simd_float4 c) {
    return simd_float4{srgb_to_linear(c.x), srgb_to_linear(c.y), srgb_to_linear(c.z), c.w};
}
inline simd_float4 to_srgb(simd_float4 c) {
    return simd_float4{linear_to_srgb(c.x), linear_to_srgb(c.y), linear_to_srgb(c.z), c.w};
}

// 0xRRGGBB (sRGB) -> linear rgba
inline simd_float4 hex(uint32_t rgb, float alpha = 1.0f) {
    return from_srgb(simd_float4{((rgb >> 16) & 0xFF) / 255.0f, ((rgb >> 8) & 0xFF) / 255.0f, (rgb & 0xFF) / 255.0f, alpha});
}

// manim's palette (manim/utils/color/manim_colors.py), linear. The plain name is the _C shade.
inline const simd_float4 WHITE = hex(0xFFFFFF), BLACK = hex(0x000000);
inline const simd_float4 GREY_A = hex(0xDDDDDD), GREY_B = hex(0xBBBBBB), GREY_C = hex(0x888888),
                             GREY_D = hex(0x444444), GREY_E = hex(0x222222), GREY = GREY_C;
inline const simd_float4 BLUE_A = hex(0xC7E9F1), BLUE_B = hex(0x9CDCEB), BLUE_C = hex(0x58C4DD),
                             BLUE_D = hex(0x29ABCA), BLUE_E = hex(0x236B8E), BLUE = BLUE_C;
inline const simd_float4 TEAL_A = hex(0xACEAD7), TEAL_B = hex(0x76DDC0), TEAL_C = hex(0x5CD0B3),
                             TEAL_D = hex(0x55C1A7), TEAL_E = hex(0x49A88F), TEAL = TEAL_C;
inline const simd_float4 GREEN_A = hex(0xC9E2AE), GREEN_B = hex(0xA6CF8C), GREEN_C = hex(0x83C167),
                             GREEN_D = hex(0x77B05D), GREEN_E = hex(0x699C52), GREEN = GREEN_C;
inline const simd_float4 YELLOW_A = hex(0xFFF1B6), YELLOW_B = hex(0xFFEA94), YELLOW_C = hex(0xF7D96F),
                             YELLOW_D = hex(0xF4D345), YELLOW_E = hex(0xE8C11C), YELLOW = YELLOW_C;
inline const simd_float4 GOLD_A = hex(0xF7C797), GOLD_B = hex(0xF9B775), GOLD_C = hex(0xF0AC5F),
                             GOLD_D = hex(0xE1A158), GOLD_E = hex(0xC78D46), GOLD = GOLD_C;
inline const simd_float4 RED_A = hex(0xF7A1A3), RED_B = hex(0xFF8080), RED_C = hex(0xFC6255),
                             RED_D = hex(0xE65A4C), RED_E = hex(0xCF5044), RED = RED_C;
inline const simd_float4 MAROON_A = hex(0xECABC1), MAROON_B = hex(0xEC92AB), MAROON_C = hex(0xC55F73),
                             MAROON_D = hex(0xA24D61), MAROON_E = hex(0x94424F), MAROON = MAROON_C;
inline const simd_float4 PURPLE_A = hex(0xCAA3E8), PURPLE_B = hex(0xB189C6), PURPLE_C = hex(0x9A72AC),
                             PURPLE_D = hex(0x715582), PURPLE_E = hex(0x644172), PURPLE = PURPLE_C;
inline const simd_float4 PINK = hex(0xD147BD), LIGHT_PINK = hex(0xDC75CD), ORANGE = hex(0xFF862F),
                             LIGHT_BROWN = hex(0xCD853F), DARK_BROWN = hex(0x8B4513), GREY_BROWN = hex(0x736357);

// --- OKLab (https://bottosson.github.io/posts/oklab/) ---

// linear rgba -> (L, a, b, alpha)
inline simd_float4 to_oklab(simd_float4 c) {
    float r = c.x, g = c.y, b = c.z;
    float l = std::cbrt(0.4122214708f * r + 0.5363325363f * g + 0.0514459929f * b);
    float m = std::cbrt(0.2119034982f * r + 0.6806995451f * g + 0.1073969566f * b);
    float s = std::cbrt(0.0883024619f * r + 0.2817188376f * g + 0.6299787005f * b);
    return simd_float4{0.2104542553f * l + 0.7936177850f * m - 0.0040720468f * s,
                       1.9779984951f * l - 2.4285922050f * m + 0.4505937099f * s,
                       0.0259040371f * l + 0.7827717662f * m - 0.8086757660f * s, c.w};
}

// (L, a, b, alpha) -> linear rgba (clamped to 0..1)
inline simd_float4 from_oklab(simd_float4 c) {
    float l = c.x + 0.3963377774f * c.y + 0.2158037573f * c.z;
    float m = c.x - 0.1055613458f * c.y - 0.0638541728f * c.z;
    float s = c.x - 0.0894841775f * c.y - 1.2914855480f * c.z;
    l = l * l * l, m = m * m * m, s = s * s * s;
    return simd_float4{std::clamp(4.0767416621f * l - 3.3077115913f * m + 0.2309699292f * s, 0.0f, 1.0f),
                       std::clamp(-1.2684380046f * l + 2.6097574011f * m - 0.3413193965f * s, 0.0f, 1.0f),
                       std::clamp(-0.0041960863f * l - 0.7034186147f * m + 1.7076147010f * s, 0.0f, 1.0f), c.w};
}

// a -> b at t in 0..1, through OKLab
inline simd_float4 lerp(simd_float4 a, simd_float4 b, float t) {
    simd_float4 la = to_oklab(a);
    return from_oklab(la + (to_oklab(b) - la) * t);
}

// The colour at t in 0..1 along evenly spaced stops (t outside 0..1 holds the end colours).
inline simd_float4 at(const std::vector<simd_float4>& stops, float t) {
    if (stops.empty()) throw std::invalid_argument("col: a gradient needs at least one colour");
    if (stops.size() == 1) return stops[0];
    float x = std::clamp(t, 0.0f, 1.0f) * (stops.size() - 1);
    size_t i = std::min((size_t)x, stops.size() - 2);
    return lerp(stops[i], stops[i + 1], x - i);
}

// --- As matrices, for material.colors and the bar charts ---

// One colour as a [4] Float matrix.
inline matrix rgba(simd_float4 c) { return matrix::of<float>({c.x, c.y, c.z, c.w}); }

// n colours evenly through the stops, first to last: [n, 4] Float, e.g. one per bar by position.
inline matrix gradient(const std::vector<simd_float4>& stops, size_m n) {
    matrix out = matrix::withShape({n, 4}, dtype::Float);
    out.begin_refcount();
    float* p = static_cast<float*>(out.buffer);
    for (size_m i = 0; i < n; i++) {
        simd_float4 c = at(stops, n > 1 ? float(i) / float(n - 1) : 0.0f);
        p[4 * i] = c.x, p[4 * i + 1] = c.y, p[4 * i + 2] = c.z, p[4 * i + 3] = c.w;
    }
    return out;
}

// Colours by value: lo -> the first stop, hi -> the last, any shape [...] -> [..., 4] Float.
// Lazy (a lookup into a 256-colour table of the gradient), so it follows values that are still
// being computed, e.g. a histogram's counts.
inline matrix map(const matrix& values, const std::vector<simd_float4>& stops, float lo, float hi) {
    constexpr size_m table_size = 256;
    matrix table = gradient(stops, table_size);
    float scale = hi > lo ? (table_size - 1) / (hi - lo) : 0.0f;
    matrix v = values.type == dtype::Float ? values : values.astype(dtype::Float);
    matrix idx = ((v - lo) * scale + 0.5f).clamp(0.0, table_size - 1.0).astype(dtype::Int32);
    return table.take(idx, 0);
}

// Same over the values' own range [min, max] (reads both back, so it evaluates them).
inline matrix map(const matrix& values, const std::vector<simd_float4>& stops) {
    matrix v = values.type == dtype::Float ? values : values.astype(dtype::Float);
    return map(v, stops, v.min().at<float>(), v.max().at<float>());
}

} // namespace col

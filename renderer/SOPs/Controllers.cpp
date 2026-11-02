#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include <simd/simd.h>
#include "../../matrix/matrix.h"
#include "GeoNode.cpp"
#include "MeshPrimitives.cpp"
// Hidden from CLion's code analysis only (real compilers don't define these): CLion Nova can't parse
// Objective-C and reads MDLTransform's `@property ... matrix;` as a global variable that shadows
// the matrix class in every file that includes this one.
#if !defined(__RESHARPER__) && !defined(__CLION_IDE__)
#import <ModelIO/ModelIO.h>
#import <MetalKit/MetalKit.h>
#endif

class PointCloudController {
public:
    GeoNodeImpl node;
    float point_size = 5.0f; // Typical size for points, mapped to line_width or similar depending on shader

    PointCloudController(std::string name) 
    {
        node = GeoNode::create(std::move(name));
        node->material.pipeline_state = 5; // Point Cloud State
        node->draggable = false;
    }

    PointCloudController(std::string name, size_t point_count) 
    {
        node = GeoNode::create(std::move(name));
        node->material.pipeline_state = 5; // Point Cloud State
        node->draggable = false;
        
        matrix x_stream = matrix::zeros({(size_m)point_count, 1}, dtype::Float);
        matrix y_stream = matrix::zeros({(size_m)point_count, 1}, dtype::Float);
        matrix z_stream = matrix::zeros({(size_m)point_count, 1}, dtype::Float);

        std::vector<matrix> streams = {x_stream, y_stream, z_stream};
        node->mesh.vert_position = matrix::concat(streams, 1);
        
        matrix indices(1, dtype::UInt32);
        indices.total_size = point_count;
        indices.shape()[0] = point_count;
        indices.calcStrides();
        indices.buffer = new uint8_t[point_count * sizeof(uint32_t)];
        uint32_t* buf = (uint32_t*)indices.buffer;
        for (size_t i = 0; i < point_count; ++i) buf[i] = (uint32_t)i;
        if (point_count > 10) indices.buildMetalBuffer();
        
        node->mesh.indices = indices;
    }

    void update(matrix points, matrix colors) {
        size_t point_count = points.shape()[0];
        
        matrix indices(1, dtype::UInt32);
        indices.total_size = point_count;
        indices.shape()[0] = point_count;
        indices.calcStrides();
        indices.buffer = new uint8_t[point_count * sizeof(uint32_t)];
        uint32_t* buf = (uint32_t*)indices.buffer;
        for (size_t i = 0; i < point_count; ++i) buf[i] = (uint32_t)i;
        if (point_count > 10) indices.buildMetalBuffer();
        
        GeoNodeImpl node_to_update = node;
        float current_point_size = point_size;
        
        dispatch_async(dispatch_get_main_queue(), ^{
            node_to_update->mesh.vert_position = points;
            node_to_update->mesh.indices = indices;
            node_to_update->material.pipeline_state = 5; // Ensure pipeline state
            node_to_update->material.colors = colors;
            node_to_update->material.line_width = current_point_size; // Some shaders use this for point size
        });
    }

    // Hides the cloud (the Viewer skips nodes with an empty mesh). Swaps in a mesh with no data and
    // no Metal buffer, on the main thread like update(); safe from any thread.
    void clear() {
        GeoNodeImpl n = node;
        dispatch_async(dispatch_get_main_queue(), ^{
            n->mesh.vert_position = matrix(0, 0, dtype::Float);
            n->mesh.indices = matrix(0, 0, dtype::UInt32);
        });
    }

//    void update(const matrix& new_x, const matrix& new_y) {
//        matrix z_stream = matrix::zeros({(size_m)new_x.shape()[0], 1}, dtype::Float);
//        std::vector<matrix> streams = {new_x, new_y, z_stream};
//        matrix points = matrix::concat(streams, 1);
//        update(points);
//    }
};

// An arbitrary triangle mesh with one colour per vertex (e.g. the foveated 2.5D grid's tiles,
// AI/FoveatedGrid.h). update()/clear() can be called from any thread: the new geometry is swapped
// in on the main thread, between frames, like PointCloudController. Hand it plain matrices (no
// tape into a graph another thread is running).
class TriangleMeshController {
public:
    GeoNodeImpl node;

    TriangleMeshController(std::string name) {
        node = GeoNode::create(std::move(name));
        node->draggable = false;
        node->material.pipeline_state = 3;   // the default mesh pipeline: indexed triangles, per-vertex colours
    }

    // vertices [V, 3] Float, colours [V, 4] Float RGBA, indices [3T] UInt32
    void update(matrix vertices, matrix colours, matrix indices) {
        // DAGNodeVertexShader declares a UV buffer (slot 5) and Metal's validation layer requires
        // every declared buffer to be bound, even though the shader copes without one - so give
        // the mesh all-zero UVs (the shader's own fallback). MeshPrimitives meshes always have UVs.
        const size_m v = vertices.shape()[0];
        matrix uvs = matrix::withShape({v, 2}, dtype::Float);
        uvs.begin_refcount();
        std::fill_n(static_cast<float*>(uvs.buffer), (size_t)v * 2, 0.0f);

        GeoNodeImpl n = node;
        dispatch_async(dispatch_get_main_queue(), ^{
            n->mesh.vert_position = vertices;
            n->mesh.uv_coords = uvs;
            n->mesh.indices = indices;
            n->material.colors = colours;
        });
    }

    // Hides it: the Viewer skips nodes with an empty mesh.
    void clear() {
        GeoNodeImpl n = node;
        dispatch_async(dispatch_get_main_queue(), ^{
            n->mesh.vert_position = matrix(0, 0, dtype::Float);
            n->mesh.indices = matrix(0, 0, dtype::UInt32);
        });
    }
};

// Wireframe boxes: the unit cube (MeshPrimitives::cube + cube_edges) on the edge pipeline (6), one
// instance per box, so any number of boxes is one draw call. Each instance's transform maps the unit
// cube (-0.5..0.5 on every axis) onto its box. Drawn twice for a sci-fi look: a thin bright core
// (`node`) and a wider translucent glow (`glow`, a child, so it shares the instances). The core is
// drawn first and the depth test is Less, so the glow only lands around the core, not over it.
// update()/clear() can be called from any thread (swapped in on the main thread, like the others).
class WireBoxController {
public:
    GeoNodeImpl node;
    GeoNodeImpl glow;

    WireBoxController(std::string name,
                      simd_float4 core_rgba = {0.72f, 0.93f, 1.0f, 1.0f},
                      simd_float4 glow_rgba = {0.25f, 0.65f, 1.0f, 0.28f},
                      float core_width = 0.0004f, float glow_width = 0.0014f) {
        cube_ = MeshPrimitives::cube(1.0f, 1.0f, 1.0f);
        edges_ = MeshPrimitives::cube_edges();

        node = GeoNode::create(name);
        glow = GeoNode::create(name + " glow");
        auto style = [&](const GeoNodeImpl& n, simd_float4 rgba, float width) {
            n->topology.edges = edges_;
            n->material.pipeline_state = 6;
            n->material.colors = matrix::of<float>({rgba.x, rgba.y, rgba.z, rgba.w}).reshape(1, 4);
            n->material.line_width = width;   // edge shader: half-width = line_width * camera distance
            n->draggable = false;
        };
        style(node, core_rgba, core_width);
        style(glow, glow_rgba, glow_width);
        node->add_child(glow);
    }

    // transforms [N, 4, 4] Float, N >= 1: unit cube -> box, column-vector convention (what the
    // edge shader applies: M * float4(p, 1)).
    void update(matrix transforms) {
        GeoNodeImpl n = node, g = glow;
        Mesh cube = cube_;
        dispatch_async(dispatch_get_main_queue(), ^{
            n->mesh = cube;
            g->mesh = cube;
            n->set_local_transform(transforms);   // also re-links the glow's world transform
        });
    }

    // Hides both (the Viewer skips nodes with an empty mesh).
    void clear() {
        GeoNodeImpl n = node, g = glow;
        dispatch_async(dispatch_get_main_queue(), ^{
            n->mesh = Mesh();
            g->mesh = Mesh();
        });
    }

private:
    Mesh cube_;
    matrix edges_ = matrix(0, 0, dtype::UInt32);
};

// A quick low-poly model made of boxes (e.g. the ego car in kitti_replay): solid unit-cube
// instances, one colour each, with a WireBoxController outline over them. The solids and the
// outline are siblings under an empty root: parenting one N-instance node under another would
// multiply them into N x N instances (see GeoNode's instance broadcasting).
// set()/clear() can be called from any thread (swapped in on the main thread).
class BoxModelController {
public:
    GeoNodeImpl node;                        // the root: add this to a viewer
    GeoNodeImpl solids;
    WireBoxController outline;

    BoxModelController(std::string name, simd_float4 edge_rgba = {0.72f, 0.93f, 1.0f, 1.0f},
                       simd_float4 glow_rgba = {0.25f, 0.65f, 1.0f, 0.28f})
        : outline(name + " outline", edge_rgba, glow_rgba) {
        node = GeoNode::create(name);
        node->draggable = false;
        solids = GeoNode::create(name + " solids");
        solids->draggable = false;
        solids->material.pipeline_state = 3;
        cube_ = MeshPrimitives::cube(1.0f, 1.0f, 1.0f);
        node->add_child(solids);
        node->add_child(outline.node);
    }

    // transforms [N, 4, 4]: unit cube -> each part (column-vector convention, like
    // WireBoxController); colours [N, 1, 4]: one RGBA per part.
    void set(matrix transforms, matrix colours) {
        GeoNodeImpl s = solids;
        Mesh cube = cube_;
        dispatch_async(dispatch_get_main_queue(), ^{
            s->mesh = cube;
            s->material.colors = colours;
            s->set_local_transform(transforms);
        });
        outline.update(transforms);
    }

    void clear() {
        GeoNodeImpl s = solids;
        dispatch_async(dispatch_get_main_queue(), ^{ s->mesh = Mesh(); });
        outline.clear();
    }

private:
    Mesh cube_;
};

// Text labels that always face the camera and keep a constant size on screen (like three.js
// sprites / CSS2D labels, or RViz's TEXT_VIEW_FACING). Glyphs come from one atlas texture (Menlo
// Bold, drawn once with CPURasteriser::drawTextCG, which MatrixH.mm includes before this file), and
// every glyph of every label is one instance of a unit quad on pipeline 7 (DAGShaders.metal
// vertex_label): one draw call for all labels. Each label sits on a translucent plate for contrast.
// The glyphs are mesh data (mesh.glyphs); the node's transform is an ordinary transform, so a label
// can be a child (add_child) and moves with its parent. Text keeps its size on screen whatever the
// transform's scale: only the anchors are transformed.
// Characters outside the atlas (lower case is upper-cased) are drawn as spaces.
// update()/clear() can be called from any thread (swapped in on the main thread).
class LabelController {
public:
    GeoNodeImpl node;
    float size = 0.018f;   // glyph height, as a fraction of the viewport height

    LabelController(std::string name) {
        node = GeoNode::create(std::move(name));
        node->draggable = false;
        node->material.pipeline_state = 7;
        node->material.has_texture = true;
        node->material.texture = atlas().texture;
        node->material.line_width = size;
        quad_.vert_position = matrix::of<float>({{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}});
        quad_.uv_coords = matrix::of<float>({{0, 0}, {1, 0}, {1, 1}, {0, 1}});
        quad_.indices = matrix::of<uint32_t>({0, 1, 2, 0, 2, 3});
        quad_.vert_position.begin_refcount();
        quad_.uv_coords.begin_refcount();
        quad_.indices.begin_refcount();
    }

    // One label per anchor (in the node's space, the label's bottom centre), all the same colours.
    void update(const std::vector<simd_float3>& anchors, const std::vector<std::string>& texts,
                simd_float4 text_rgba = {0.72f, 0.93f, 1.0f, 1.0f},
                simd_float4 plate_rgba = {0.02f, 0.08f, 0.14f, 0.55f}) {
        update(anchors, texts, std::vector<float>(), text_rgba, plate_rgba);
    }

    // The same, with each label moved `lines[i]` label heights up from the anchor (negative: down), so
    // several labels can stack on one point and stay stacked at any zoom. Missing entries are 0.
    void update(const std::vector<simd_float3>& anchors, const std::vector<std::string>& texts, const std::vector<float>& lines,
                simd_float4 text_rgba = {0.72f, 0.93f, 1.0f, 1.0f},
                simd_float4 plate_rgba = {0.02f, 0.08f, 0.14f, 0.55f}) {
        const Atlas& at = atlas();
        const size_t n = std::min(anchors.size(), texts.size());
        size_t count = 0;
        for (size_t i = 0; i < n; ++i) count += 1 + texts[i].size();
        if (count == 0) { clear(); return; }

        const float advance = at.glyph_aspect, pad = 0.2f, lift = 0.35f;
        matrix glyphs = matrix::withShape({(size_m)count, 16}, dtype::Float);
        glyphs.begin_refcount();
        simd_float4* g = (simd_float4*)glyphs.buffer;   // the Glyph layout in DAGShaders.metal
        auto put = [&](simd_float3 a, float layer, simd_float4 rect, simd_float4 uv, simd_float4 rgba) {
            *g++ = simd_make_float4(a, layer);
            *g++ = rect;
            *g++ = uv;
            *g++ = rgba;
        };
        for (size_t i = 0; i < n; ++i) {
            const std::string& s = texts[i];
            const float w = advance * s.size();
            const float line_height = 1 + 2 * pad + 0.15f;   // plate plus a small gap
            const float x0 = -0.5f * w, y0 = lift + (i < lines.size() ? lines[i] : 0.0f) * line_height;
            put(anchors[i], 0, simd_make_float4(x0 - pad, y0 - pad, w + 2 * pad, 1 + 2 * pad), at.uv(Atlas::kSolid), plate_rgba);
            for (size_t c = 0; c < s.size(); ++c) {
                put(anchors[i], 1, simd_make_float4(x0 + c * advance, y0, advance, 1), at.uv(at.index(s[c])), text_rgba);
            }
        }

        GeoNodeImpl nd = node;
        Mesh quad = quad_;
        float sz = size;
        dispatch_async(dispatch_get_main_queue(), ^{
            nd->mesh = quad;
            nd->mesh.glyphs = glyphs;
            nd->material.line_width = sz;
        });
    }

    // Hides it (the Viewer skips nodes with an empty mesh).
    void clear() {
        GeoNodeImpl nd = node;
        dispatch_async(dispatch_get_main_queue(), ^{ nd->mesh = Mesh(); });
    }

private:
    Mesh quad_;

    struct Atlas {
        static constexpr const char* kChars = " 0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ./-:%()#+";
        static constexpr int kCols = 16, kCellW = 32, kCellH = 64;
        static constexpr int kSolid = 0;      // index of the filled cell (the plate), after the chars
        int rows = 0, count = 0;
        float glyph_aspect = (float)kCellW / kCellH;
        id<MTLTexture> texture = nil;

        int index(char ch) const {
            if (ch >= 'a' && ch <= 'z') ch = (char)(ch - 'a' + 'A');
            const char* p = std::strchr(kChars, ch);
            return (p && ch) ? (int)(p - kChars) + 1 : 1;   // unknown -> the space cell
        }
        // Atlas rect of cell i, inset half a texel so linear filtering doesn't bleed across cells.
        simd_float4 uv(int i) const {
            const float W = kCols * kCellW, H = rows * kCellH;
            int col = i % kCols, row = i / kCols;
            return simd_make_float4((col * kCellW + 0.5f) / W, (row * kCellH + 0.5f) / H,
                                    ((col + 1) * kCellW - 0.5f) / W, ((row + 1) * kCellH - 0.5f) / H);
        }
    };

    // Cell 0 is solid (the plates), cells 1.. are kChars. White glyphs, so only alpha matters.
    static const Atlas& atlas() {
        static const Atlas a = [] {
            Atlas at;
            at.count = 1 + (int)std::strlen(Atlas::kChars);
            at.rows = (at.count + Atlas::kCols - 1) / Atlas::kCols;
            const int W = Atlas::kCols * Atlas::kCellW, H = at.rows * Atlas::kCellH;
            matrix img = matrix::withShape({(size_m)H, (size_m)W, 4}, dtype::UInt8);
            img.begin_refcount();
            std::memset(img.buffer, 0, (size_t)W * H * 4);
            uint8_t* px = (uint8_t*)img.buffer;
            for (int y = 0; y < Atlas::kCellH; ++y)
                std::memset(px + (size_t)y * W * 4, 255, Atlas::kCellW * 4);   // cell 0
            matrix white = matrix{255u, 255u, 255u, 255u};
            for (int i = 1; i < at.count; ++i) {
                char s[2] = {Atlas::kChars[i - 1], 0};
                if (s[0] == ' ') continue;
                int cx = (i % Atlas::kCols) * Atlas::kCellW + Atlas::kCellW / 2;
                int cy = (i / Atlas::kCols) * Atlas::kCellH + Atlas::kCellH / 2;
                CPURasteriser::drawTextCG(img, s, "Menlo-Bold", 46.0f, cx, cy, white, simd_make_float2(0, 0));
            }
            at.texture = img.ToMTLTexture();
            return at;
        }();
        return a;
    }
};

// Wireframe boxes, each with a camera-facing label on top (e.g. every tracked car and its speed):
// a WireBoxController and a LabelController as siblings under an empty root, like
// BoxModelController (parenting the labels under the N-instance box node would apply every box's
// transform to every label). Each label sits on its box's top centre, the unit cube's (0, 0.5, 0)
// carried through the box's transform, so it follows any box placement.
// update()/clear() can be called from any thread (swapped in on the main thread).
class LabelledBoxController {
public:
    GeoNodeImpl node;                        // the root: add this to a viewer
    WireBoxController boxes;
    LabelController labels;
    simd_float4 text_rgba;

    LabelledBoxController(std::string name, simd_float4 edge_rgba = {0.72f, 0.93f, 1.0f, 1.0f},
                          simd_float4 glow_rgba = {0.25f, 0.65f, 1.0f, 0.28f})
        : boxes(name + " boxes", edge_rgba, glow_rgba), labels(name + " labels"), text_rgba(edge_rgba) {
        node = GeoNode::create(name);
        node->draggable = false;
        node->add_child(boxes.node);
        node->add_child(labels.node);
    }

    // transforms [N, 4, 4] Float: unit cube -> each box (column-vector convention, as
    // WireBoxController); texts: one label per box, in the same order. N = 0 clears both.
    void update(matrix transforms, const std::vector<std::string>& texts) {
        const size_m n = transforms.dims ? transforms.shape()[0] : 0;
        if (n == 0) { clear(); return; }
        std::vector<simd_float3> anchors((size_t)n);
        for (size_m i = 0; i < n; ++i) {
            const simd_float4 top = simd_mul(transforms.SIMD_MAT((int)i), simd_make_float4(0, 0.5f, 0, 1));
            anchors[i] = top.xyz;
        }
        boxes.update(transforms);
        labels.update(anchors, texts, text_rgba);
    }

    void clear() {
        boxes.clear();
        labels.clear();
    }
};

class TriangleController {
public:
    GeoNodeImpl node;

    TriangleController(std::string name, matrix anchor = matrix(0, 0, dtype::Float)) {
        node = GeoNode::create(std::move(name));
        node->mesh = MeshPrimitives::triangle(1.0f, 1.0f);
        if (anchor.total_size > 0) {
            node->mesh.vert_position = node->mesh.vert_position + anchor;
        }
    }
};

class QuadController {
public:
    GeoNodeImpl node;
    QuadController() {}
    QuadController(std::string name, matrix anchor = matrix(0, 0, dtype::Float)) {
        node = GeoNode::create(std::move(name));
        node->mesh = MeshPrimitives::quad(1.0f, 1.0f);
        if (anchor.total_size > 0) {
            node->mesh.vert_position = node->mesh.vert_position + anchor;
        }
    }

    void update(simd_float3 dir, simd_float3 origin, float thickness) {
        simd_float3 d = simd_normalize(dir);
        simd_float3 up = {0.0f, 1.0f, 0.0f};
        if (std::abs(d.y) > 0.99f) {
            up = {1.0f, 0.0f, 0.0f};
        }
        simd_float3 right = simd_normalize(simd_cross(up, d));
        
        simd_float3 v0 = origin - right * (thickness * 0.5f);
        simd_float3 v1 = origin + right * (thickness * 0.5f);
        simd_float3 v2 = origin + dir + right * (thickness * 0.5f);
        simd_float3 v3 = origin + dir - right * (thickness * 0.5f);
        
        node->mesh.vert_position = matrix::of<float>({
            {v0.x, v0.y, v0.z},
            {v1.x, v1.y, v1.z},
            {v2.x, v2.y, v2.z},
            {v3.x, v3.y, v3.z}
        });
        simd_float3 nrm = simd_normalize(simd_cross(v1 - v0, v3 - v0));   // the quad's new plane
        node->mesh.normals = MeshPrimitives::constant_normals(4, nrm);
    }
};

// An image on a quad, through the normal mesh path (no special pipeline): the texture rides on
// the material, like PlotterViewer's plot. The quad is 1 unit tall and W/H wide, centred on the
// origin. update() is safe from any thread: the swap lands on the main queue, between frames.
class ImageController {
public:
    enum class Order { RGBA, BGRA };   // RGBA is what matrix::fromImage gives, BGRA what CapReader does

    GeoNodeImpl node;

    ImageController(std::string name) {
        node = GeoNode::create(std::move(name));
        node->mesh = MeshPrimitives::quad(1.0f, 1.0f);
        // Row 0 at the top, like imshow: the quad's UVs put v = 0 (the texture's first row in
        // Metal) at the bottom, so flip v.
        node->mesh.uv_coords = matrix::of<float>({
            {0.0f, 1.0f},   // bottom-left
            {1.0f, 1.0f},   // bottom-right
            {1.0f, 0.0f},   // top-right
            {0.0f, 0.0f}    // top-left
        });
        node->mesh.uv_coords.begin_refcount();
        node->draggable = false;
    }

    // img: [H, W, 3] or [H, W, 4], UInt8, or Float in [0, 1]. A 3-channel image is opaque.
    void update(matrix img, Order order = Order::RGBA) {
        if (img.dims != 3 || (img.shape()[2] != 3 && img.shape()[2] != 4)) {
            throw std::runtime_error("ImageController::update: expected [H, W, 3] or [H, W, 4]");
        }
        size_m H = img.shape()[0], W = img.shape()[1];
        if (H == 0 || W == 0) return;
        // img is our own copy: if it's a view, drop the flag so the assignments below rebind
        // it instead of copying into the caller's buffer.
        img.flags &= ~NON_OWNERSHIP_FLAG;
        if (img.type != dtype::UInt8) img = (img.clamp(0.0f, 1.0f) * 255.0f).astype(dtype::UInt8);

        // ToMTLTexture makes a BGRA texture: lay the channels out as B, G, R, A.
        matrix bgra = img;
        if (order == Order::RGBA || img.shape()[2] == 3) {
            matrix alpha = img.shape()[2] == 4 ? img[R(), R(), R(3, 4)]
                                               : (matrix::ones({H, W, 1}, dtype::UInt8) * 255).astype(dtype::UInt8);
            matrix b = order == Order::RGBA ? img[R(), R(), R(2, 3)] : img[R(), R(), R(0, 1)];
            matrix r = order == Order::RGBA ? img[R(), R(), R(0, 1)] : img[R(), R(), R(2, 3)];
            bgra = matrix::concat({b, img[R(), R(), R(1, 2)], r, alpha}, 2);
        }
        if (bgra.tape) bgra.eval_metal();

        // Same size as the texture already on the node: blit into it, nothing else changes.
        id<MTLTexture> current = node->material.texture;
        if (current && current.width == (NSUInteger)W && current.height == (NSUInteger)H) {
            bgra.CopyToTexture(current);
            return;
        }
        id<MTLTexture> tex = bgra.ToMTLTexture();

        // A fresh [1, 4, 4] buffer (as Comp's transform_of) rather than SIMD_MAT into the current
        // one, which a frame still in flight may be reading.
        matrix aspect = matrix::eye(4).unsqueeze(0);
        aspect.eval();
        aspect.releaseTape();
        aspect.SIMD_MAT(0).columns[0][0] = (float)W / (float)H;

        GeoNodeImpl n = node;
        dispatch_async(dispatch_get_main_queue(), ^{
            n->material.texture = tex;
            n->material.has_texture = true;
            n->set_local_transform(aspect);
        });
    }
};

class CubeController {
public:
    GeoNodeImpl node;

    CubeController(std::string name, matrix anchor = matrix(0, 0, dtype::Float)) {
        node = GeoNode::create(std::move(name));
        node->mesh = MeshPrimitives::cube(1.0f, 1.0f, 1.0f);
        if (anchor.total_size > 0) {
            node->mesh.vert_position = node->mesh.vert_position + anchor;
        }
    }
};

class CircleController {
public:
    GeoNodeImpl node;

    CircleController(std::string name, matrix anchor = matrix(0, 0,dtype::Float)) {
        node = GeoNode::create(std::move(name));
        node->mesh = MeshPrimitives::circle(1.0f, 32);
        if (anchor.total_size > 0) {
            node->mesh.vert_position = node->mesh.vert_position + anchor;
        }
    }
};

class SphereController {
public:
    GeoNodeImpl node;

    SphereController(std::string name, matrix anchor = matrix(0, 0, dtype::Float)) {
        node = GeoNode::create(std::move(name));
        node->mesh = MeshPrimitives::sphere(1.0f, 16, 16);
        if (anchor.total_size > 0) {
            node->mesh.vert_position = node->mesh.vert_position + anchor;
        }
    }
};

class CylinderController {
public:
    GeoNodeImpl node;

    CylinderController(std::string name, float radius = 1.0f, float length = 1.0f, int subdivisions = 32, matrix anchor = matrix(0, 0, dtype::Float)) {
        node = GeoNode::create(std::move(name));
        node->mesh = MeshPrimitives::cylinder(radius, length, subdivisions);
        if (anchor.total_size > 0) {
            node->mesh.vert_position = node->mesh.vert_position + anchor;
        }
    }
};

class ConeController {
public:
    GeoNodeImpl node;

    ConeController(std::string name, float radius = 1.0f, float length = 1.0f, int subdivisions = 32, matrix anchor = matrix(0, 0, dtype::Float)) {
        node = GeoNode::create(std::move(name));
        node->mesh = MeshPrimitives::cone(radius, length, subdivisions);
        if (anchor.total_size > 0) {
            node->mesh.vert_position = node->mesh.vert_position + anchor;
        }
    }
    
    // Re-points the cone: base at the origin, tip at `dir`, base radius r. Rebuilds the mesh (fresh buffers,
    // normals included) rather than writing into buffers an in-flight frame might be reading.
    void update(simd_float3 dir, float r) {
        const int n = (int)(node->mesh.vert_position.shape()[0] - 2) / 3;   // MeshPrimitives::cone has 3n + 2 vertices
        const float len = simd_length(dir);
        if (n < 3 || len <= 0.0f) return;
        simd_float3 axis = dir / len;
        simd_float3 up = {0.0f, 1.0f, 0.0f};
        // Avoid singularity if dir is parallel to up
        if (std::abs(axis.y) > 0.99f) {
            up = {1.0f, 0.0f, 0.0f};
        }
        simd_float3 right = simd_normalize(simd_cross(up, axis));
        up = -simd_normalize(simd_cross(right, axis));

        // Build it along +y, then rotate +x -> right, +y -> axis, +z -> up (positions and normals alike).
        Mesh m = MeshPrimitives::cone(r, len, n);
        for (matrix* a : {&m.vert_position, &m.normals}) {
            float* v = static_cast<float*>(a->buffer);
            for (int i = 0; i < 3 * n + 2; i++) {
                simd_float3 q = right * v[3 * i] + axis * v[3 * i + 1] + up * v[3 * i + 2];
                v[3 * i] = q.x, v[3 * i + 1] = q.y, v[3 * i + 2] = q.z;
            }
        }
        node->mesh = m;
    }
};

class GizmoController {
public:
    GeoNodeImpl node; // Root node
    GeoNodeImpl x_axis;
    GeoNodeImpl y_axis;
    GeoNodeImpl z_axis;
    GeoNodeImpl xy_plane;
    GeoNodeImpl yz_plane;
    GeoNodeImpl zx_plane;

    GizmoController(std::string name = "Gizmo") {
        node = GeoNode::create(std::move(name));

        x_axis = GeoNode::create("X_Axis");
        y_axis = GeoNode::create("Y_Axis");
        z_axis = GeoNode::create("Z_Axis");
        xy_plane = GeoNode::create("XY_Plane");
        yz_plane = GeoNode::create("YZ_Plane");
        zx_plane = GeoNode::create("ZX_Plane");

        Mesh x_mesh = MeshPrimitives::cylinder(0.05f, 1.0f, 16);
        Mesh y_mesh = MeshPrimitives::cylinder(0.05f, 1.0f, 16);
        Mesh z_mesh = MeshPrimitives::cylinder(0.05f, 1.0f, 16);
        Mesh quad_mesh = MeshPrimitives::quad(0.3f, 0.3f);

        x_axis->mesh = x_mesh;
        y_axis->mesh = y_mesh;
        z_axis->mesh = z_mesh;
        xy_plane->mesh = quad_mesh;
        yz_plane->mesh = quad_mesh;
        zx_plane->mesh = quad_mesh;

        x_axis->material.colors = matrix::of<float>({{1.0f, 0.0f, 0.0f, 1.0f}}); // Red
        y_axis->material.colors = matrix::of<float>({{0.0f, 1.0f, 0.0f, 1.0f}}); // Green
        z_axis->material.colors = matrix::of<float>({{0.0f, 0.0f, 1.0f, 1.0f}}); // Blue
        xy_plane->material.colors = matrix::of<float>({{0.0f, 0.0f, 1.0f, 0.6f}}); // Yellow
        yz_plane->material.colors = matrix::of<float>({{1.0f, 0.0f, 0.0f, 0.6f}}); // Cyan
        zx_plane->material.colors = matrix::of<float>({{0.0f, 1.0f, 0.0f, 0.6f}}); // Magenta
        
        DepthBias gizmo_bias = {false, 0.0f, 0.0f, 0.0f};
        x_axis->material.depth_bias = gizmo_bias;
        y_axis->material.depth_bias = gizmo_bias;
        z_axis->material.depth_bias = gizmo_bias;
        xy_plane->material.depth_bias = gizmo_bias;
        yz_plane->material.depth_bias = gizmo_bias;
        zx_plane->material.depth_bias = gizmo_bias;

        x_axis->local_transform.SIMD_MAT(0) = RotationZ(-90);
        y_axis->local_transform.SIMD_MAT(0) = Identity();
        z_axis->local_transform.SIMD_MAT(0) = RotationX(90);

        xy_plane->local_transform.SIMD_MAT(0) = Translation(simd_make_float3(0.3f, 0.3f, 0.0f));
        yz_plane->local_transform.SIMD_MAT(0) = Translation(simd_make_float3(0.0f, 0.3f, -0.3f)) * RotationY(-90);
        zx_plane->local_transform.SIMD_MAT(0) = Translation(simd_make_float3(0.3f, 0.0f, -0.3f)) * RotationX(90);

        node->add_child(x_axis);
        node->add_child(y_axis);
        node->add_child(z_axis);
        node->add_child(xy_plane);
        node->add_child(yz_plane);
        node->add_child(zx_plane);
    }
};

class LineController {
public:
    GeoNodeImpl node;
    float line_width = 0.0025f;
    bool round_joins = false;   // pipeline 8 (round joins and caps, 12 vertices per point) instead of 4 (strip, 2)

    LineController(std::string name = "Line3D", float width = 0.0025f) : line_width(width) {
        node = GeoNode::create(std::move(name));
        node->material.pipeline_state = 4;
    }
    // Round joins and caps (pipeline 8) or the cheap mitered strip (pipeline 4, sharp turns spike).
    // Applied on the main queue like update(), and kept by later updates.
    void set_round_joins(bool on) {
        round_joins = on;
        GeoNodeImpl node_to_update = node;
        dispatch_async(dispatch_get_main_queue(), ^{
            node_to_update->material.pipeline_state = on ? 8 : 4;
        });
    }
// legacy bout to be gone
//    void update(const matrix& points, matrix cam_forward_matrix) {
//        if (points.shape()[0] < 2) return;
//
//        // Static Heavy Math (Runs once, cached in the DAG)
//        matrix direction_vectors = points.slice(R(1, points.shape()[0]), 0) - points.slice(R(0, points.shape()[0] - 1), 0);
//        matrix norm_direction_vectors = direction_vectors / matrix::sqrt((direction_vectors * direction_vectors).sum(-1, true));
//        
//        matrix pool_norm_direction_vectors = norm_direction_vectors.slice(R(1, points.shape()[0] - 1), 0) + norm_direction_vectors.slice(R(0, points.shape()[0] - 2), 0);
//        matrix pool_mag_sq = ((pool_norm_direction_vectors * pool_norm_direction_vectors).sum(-1, true));
//        matrix norm_pool = 2.0f * pool_norm_direction_vectors / pool_mag_sq;
//        
//        norm_direction_vectors = matrix::concat({norm_direction_vectors[R(0, 1)], norm_pool, norm_direction_vectors[R(-1, LLONG_MAX)]}, 0);
//        
//        // Dynamic Math (Runs every frame when cam_forward_matrix is invalidated)
//        matrix perp_vectors = matrix::cross(cam_forward_matrix, norm_direction_vectors);
//        perp_vectors =( perp_vectors / matrix::sqrt((perp_vectors * perp_vectors).sum(-1, true)) ) * matrix::sqrt((norm_direction_vectors * norm_direction_vectors).sum(-1, true));
//        matrix layer1 = points + perp_vectors * line_width;
//        matrix layer2 = points - perp_vectors * line_width;
//        matrix inter_leave = matrix::concat({layer1, layer2}, -1).reshape(2 * points.shape()[0], points.shape()[1]);
//        
//        
//        // Generate indices for the quad strip
//        uint32_t num_points = points.shape()[0];
//        uint32_t num_quads = num_points - 1;
//        uint32_t num_indices = num_quads * 6;
//        
//        matrix indices = matrix::zeros({(size_m)num_indices}, dtype::UInt32);
//        uint32_t* idx_buf = (uint32_t*)indices.buffer;
//        for (uint32_t i = 0; i < num_quads; ++i) {
//            idx_buf[i*6 + 0] = i*2 + 1;
//            idx_buf[i*6 + 1] = i*2 + 0;
//            idx_buf[i*6 + 2] = i*2 + 2;
//            idx_buf[i*6 + 3] = i*2 + 1;
//            idx_buf[i*6 + 4] = i*2 + 2;
//            idx_buf[i*6 + 5] = i*2 + 3;
//        }
//        indices.buildMetalBuffer();
//        
//        
//        // Capture the shared_ptr explicitly so it stays alive even if the GraphViewer is destroyed.
//        GeoNodeImpl node_to_update = node;
//        
//        // Dispatch the mesh assignment to the main queue to avoid a data race
//        // with the main rendering thread accessing graph_node->mesh during Viewer::draw()
//        dispatch_async(dispatch_get_main_queue(), ^{
//            node_to_update->mesh.vert_position = inter_leave;
//            node_to_update->mesh.indices = indices;
//        });
//    }

    // Any dtype is promoted to Float (the line shader reads float positions).
    void update(matrix points) {
        if (points.shape()[0] < 2) return;
        if (points.type != dtype::Float) points = points.astype(dtype::Float);
        
        GeoNodeImpl node_to_update = node;
        float current_line_width = line_width;
        uint32_t state = round_joins ? 8 : 4;
        
        dispatch_async(dispatch_get_main_queue(), ^{
            node_to_update->mesh.vert_position = points; // Send raw points directly!
            node_to_update->mesh.indices = matrix(0, 0, dtype::UInt32); // Clear indices, the vertex shader builds the line
            node_to_update->material.pipeline_state = state; // 4: strip, 8: round joins
            node_to_update->material.line_width = current_line_width;
        });
    }
    
    
    void update(matrix x, matrix y) {
        if (x.type != dtype::Float) x = x.astype(dtype::Float);
        if (y.type != dtype::Float) y = y.astype(dtype::Float);
        matrix points = matrix::stack({x, y, x.zeros()}, -1);
        
        GeoNodeImpl node_to_update = node;
        float current_line_width = line_width;
        uint32_t state = round_joins ? 8 : 4;
        
        dispatch_async(dispatch_get_main_queue(), ^{
            node_to_update->mesh.vert_position = points; // Send raw points directly!
            node_to_update->mesh.indices = matrix(0, 0, dtype::UInt32); // Clear indices, the vertex shader builds the line
            node_to_update->material.pipeline_state = state; // 4: strip, 8: round joins
            node_to_update->material.line_width = current_line_width;
        });
    }
};

inline float wrapf(float x, float y) {
    float r = std::fmod(x, y);
    return r < 0.0f ? r + y : r;
}

inline float wrap_centered(float x, float y) {
    return x - std::round(x / y) * y;
}

class GridController {
public:
    GeoNodeImpl root_node;
//    LineController major_lines;
//    LineController minor_lines;
    
    GeoNodeImpl major_lines;
    GeoNodeImpl minor_lines;
    uint32_t num_major_lines = 31;
    uint32_t num_minor_lines;
    
    // Cache to prevent invalidating the grid every frame
    simd_float3 last_cam_pos = {1e9f, 1e9f, 1e9f};
    float last_zoom_factor = 1e9f;
    simd_float3 last_graph_scale = {1e9f, 1e9f, 1e9f};
    simd_float3 last_graph_offset = {1e9f, 1e9f, 1e9f};
    
    GridController() : major_lines(GeoNode::create("MajorGrid")), minor_lines(GeoNode::create("MinorGrid")) {
        num_minor_lines = 1 + 5 * (num_major_lines-1) + 4 * 2;
        root_node = GeoNode::create("GridSystem");
        major_lines->mesh = MeshPrimitives::quad(1, 1.0f);
        major_lines->material.colors = matrix::repeating({2 * num_major_lines}, {1.0f, 1.0f, 1.0f, 1.0f});
        major_lines->material.colors = major_lines->material.colors.unsqueeze(1);
        major_lines->material.colors.eval_cpu();
        major_lines->material.colors.shape()[1] = 4;
        major_lines->material.colors.strides()[1] = 0;

        minor_lines->mesh = MeshPrimitives::quad(1.f, 1.0f);
         major_lines->local_transform = matrix::repeating({2 * num_major_lines}, matrix::eye(4));
//        major_lines->local_transform = matrix::zeros({2 * num_major_lines, 4, 4}, dtype::Float);
        minor_lines->local_transform = matrix::repeating({2 * num_minor_lines}, matrix::eye(4));
        major_lines->local_transform.make_leaf();
        minor_lines->local_transform.make_leaf();
        root_node->add_child(major_lines);
        root_node->add_child(minor_lines);
    }

    void get_dynamic_cell_sizes(float zoom_factor, float& big_cell, float& small_cell) {
        // Desmos-style 1-2-5 LOD scaling for infinite grids.
        // This ensures the grid scales up more frequently so we never run out of lines.
        float log_zoom = std::log10(std::fmax(zoom_factor, 0.001f));
        float order = std::floor(log_zoom);
        float scale = std::pow(10.0f, order);

        float normalized_zoom = zoom_factor / scale; // Always in [1, 10)

        float multiplier;
        if (normalized_zoom < 2.0f) {
            multiplier = 1.0f;
        } else if (normalized_zoom < 5.0f) {
            multiplier = 2.0f;
        } else {
            multiplier = 5.0f;
        }

        big_cell = multiplier * scale * big_cell;
        small_cell = big_cell / 5.0f; // Maintains exactly 5 minor subdivisions per major cell
    }

    void update_grid(simd_float3 cam_pos, float zoom_factor, simd_float3 graph_scale, simd_float3 graph_offset) {
        if (simd_distance(last_cam_pos, cam_pos) < 1e-4f && 
            std::abs(last_zoom_factor - zoom_factor) < 1e-4f &&
            simd_distance(last_graph_scale, graph_scale) < 1e-4f &&
            simd_distance(last_graph_offset, graph_offset) < 1e-4f) {
            return;
        }
        last_cam_pos = cam_pos;
        last_zoom_factor = zoom_factor;
        last_graph_scale = graph_scale;
        last_graph_offset = graph_offset;

        // TODO (Aditya): Logic for dynamic grid generation
        float base_thickness = 0.001f / 4;

        // Counteract the grid's independent world status by scaling thickness purely by camera zoom
        float final_thickness = base_thickness * zoom_factor;

        // The true visual zoom of the grid relative to the screen is affected by both the camera distance
        // AND the user's manual zooming (which scales the graph_node).
        // We calibrate the zoom_factor (camera distance) by ~11.0 so the baseline grid size is preserved.
        float cal_zoom = zoom_factor / 11.0f;
        // Only guard against 0: data spanning ~1e9 has a scale ~1e-9, and clamping it higher picks
        // cells far too small for it, which stacks every line on one spot.
        float effective_zoom_x = cal_zoom / std::fmax(graph_scale.x, FLT_MIN);
        float effective_zoom_y = cal_zoom / std::fmax(graph_scale.y, FLT_MIN);

        float big_cell_width_x, small_cell_width_x;
        big_cell_width_x = 2.0;
        small_cell_width_x = 2.0 / 5.0;
        get_dynamic_cell_sizes(effective_zoom_x, big_cell_width_x, small_cell_width_x);

        float big_cell_width_y, small_cell_width_y;
        big_cell_width_y = 2.0;
        small_cell_width_y = 2.0 / 5.0;
        get_dynamic_cell_sizes(effective_zoom_y, big_cell_width_y, small_cell_width_y);

        float big_cell_width_x_ws = big_cell_width_x * graph_scale.x;
        float small_cell_width_x_ws = small_cell_width_x * graph_scale.x;

        float big_cell_width_y_ws = big_cell_width_y * graph_scale.y;
        float small_cell_width_y_ws = small_cell_width_y * graph_scale.y;

        int offset_from_centre_x = graph_offset.x / big_cell_width_x_ws;
        int offset_from_centre_y = graph_offset.y / big_cell_width_y_ws;

        major_lines->material.colors = matrix::repeating({2 * num_major_lines}, {1.0f, 1.0f, 1.0f, 1.0f});
        major_lines->material.colors = major_lines->material.colors.unsqueeze(1);
        major_lines->material.colors.eval_cpu();
        major_lines->material.colors.shape()[1] = 4;
        major_lines->material.colors.strides()[1] = 0;
        // 1. Calculate the visible window in Graph Space
         for (int i = 0; i < num_major_lines; i ++) {
             float x =  big_cell_width_x * (i - (int)(num_major_lines * 0.5f)) ;
             int off = (num_major_lines / 2) + offset_from_centre_x;
             // To make the grid follow the graph, you apply the graph's translation and scale mathematically
             simd_float3 world_pos = simd_make_float3(x * graph_scale.x + fmod(graph_offset.x, big_cell_width_x_ws), 0, 0);

             if (i==off) {
                float* colors_ptr = (float*)major_lines->material.colors.buffer;
                colors_ptr[i * 4 + 0] = 0.0f;
                colors_ptr[i * 4 + 1] = 1.0f;
                colors_ptr[i * 4 + 2] = 0.0f;
                colors_ptr[i * 4 + 3] = 1.0f;
                major_lines->local_transform.SIMD_MAT(i) = Translation(world_pos) * Scale(simd_make_float3(final_thickness * 10, 1000.0f * zoom_factor, 1));
            } else {
                major_lines->local_transform.SIMD_MAT(i) = Translation(world_pos) * Scale(simd_make_float3(final_thickness * 3, 1000.0f * zoom_factor, 1));
            }

         }

        for (int i = 0; i < num_minor_lines; i ++) {
            float x =  small_cell_width_x * (i - (int)(num_minor_lines * 0.5f)) ;

            // To make the grid follow the graph, you apply the graph's translation and scale mathematically
            simd_float3 world_pos = simd_make_float3(x * graph_scale.x + fmod(graph_offset.x, small_cell_width_x_ws), 0, 0);
            minor_lines->local_transform.SIMD_MAT(i) = Translation(world_pos) * Scale(simd_make_float3(final_thickness, 1000.0f * zoom_factor, 1));

        }


         for (int i = 0; i < num_major_lines; i ++) {
             float y =  big_cell_width_y * (i - (int)(num_major_lines * 0.5f)) ;
             int off_y = (num_major_lines / 2) + offset_from_centre_y;

             // To make the grid follow the graph, you apply the graph's translation and scale mathematically
             simd_float3 world_pos = simd_make_float3(0, y * graph_scale.y + fmod(graph_offset.y, big_cell_width_y_ws), 0);

             if (i == off_y) {
                 float* colors_ptr = (float*)major_lines->material.colors.buffer;
                 colors_ptr[(num_major_lines + i) * 4 + 0] = 1.0f;
                 colors_ptr[(num_major_lines + i) * 4 + 1] = 0.0f;
                 colors_ptr[(num_major_lines + i) * 4 + 2] = 0.0f;
                 colors_ptr[(num_major_lines + i) * 4 + 3] = 1.0f;
                 major_lines->local_transform.SIMD_MAT(num_major_lines+i) = Translation(world_pos) * Scale(simd_make_float3(1000.0f * zoom_factor, final_thickness * 10, 1));
             } else {
                 major_lines->local_transform.SIMD_MAT(num_major_lines+i) = Translation(world_pos) * Scale(simd_make_float3(1000.0f * zoom_factor, final_thickness * 3, 1));
             }
         }

        for (int i = 0; i < num_minor_lines; i ++) {
            float y =  small_cell_width_y * (i - (int)(num_minor_lines * 0.5f)) ;

            // To make the grid follow the graph, you apply the graph's translation and scale mathematically
            simd_float3 world_pos = simd_make_float3(0, y * graph_scale.y + fmod(graph_offset.y, small_cell_width_y_ws), 0);

            minor_lines->local_transform.SIMD_MAT(num_minor_lines+i) = Translation(world_pos) * Scale(simd_make_float3(1000.0f * zoom_factor, final_thickness, 1));
        }

        major_lines->local_transform.tape->version = ++global_epoch;
        minor_lines->local_transform.tape->version = ++global_epoch;
        
        // 2. Determine major/minor step size based on zoom_factor
        // 3. Generate matrix of points for major lines
        // 4. Generate matrix of points for minor lines
        
        // Example boilerplate for pushing points:
        // major_lines.update(major_points);
        // minor_lines.update(minor_points);
    }
};

//#pragma once
//#include "Viewer.cpp"
//#include "Controllers.cpp"
//#include "matrix.h"
//@import Utils;
//
//
//class GraphViewer : public Viewer {
//public:
//    GeoNodeImpl graph_node;
//    GeoNodeImpl graph_frame;
//    GeoNodeImpl X_axis;
//    GeoNodeImpl Y_axis;
//    GridController grid;
//
//    GraphViewer(simd::float3 centre = {0, 0, 0}, simd::float3 size = {2, 2, 200}) {
//        simd::float3 clip_min = centre - size/2;
//        simd::float3 clip_max = centre + size/2;
//
//        X_axis = GeoNode::create("X axis");
//        X_axis->mesh = MeshPrimitives::quad(1.0f, 1.0f); // Base 1x1 quad, we will scale it dynamically!
//        X_axis->material.colors = {1.0f, 0.0f, 0.0f, 1.0f}; // red
//        X_axis->material.clip_min = clip_min;
//        X_axis->material.clip_max = clip_max;
//        nodes.push_back(X_axis);
//
//        Y_axis = GeoNode::create("Y axis");
//        Y_axis->mesh = MeshPrimitives::quad(1.0f, 1.0f);
//        Y_axis->material.colors = {0.0f, 1.0f, 0.0f, 1.0f};
//        Y_axis->material.clip_min = clip_min;
//        Y_axis->material.clip_max = clip_max;
//        nodes.push_back(Y_axis);
//
//        graph_frame = GeoNode::create("graph_frame");
//        graph_frame->mesh = MeshPrimitives::quad(size.x, size.y);
//        graph_frame->topology.edges = MeshPrimitives::quad_edges();
//        graph_frame->material.colors = {1.0f, 1.0f, 1.0f, 1.0f};
//        graph_frame->material.pipeline_state = 6;
//        graph_frame->material.line_width = 0.005f;
//        graph_frame->draggable = false;
//        nodes.push_back(graph_frame);
//
//        graph_node = GeoNode::create("graph");
//        graph_node->material.clip_min = clip_min;
//        graph_node->material.clip_max = clip_max;
//        graph_node->draggable = false;
//        nodes.push_back(graph_node);
//
//        // Add the grid to the graph node so it inherits clipping
//        graph_node->add_child(grid.root_node);
//    }
//
//    void add_child(GeoNodeImpl node, simd_float3 offset) {
//        // Apply the offset in the graph's local space
//        node->local_transform.SIMD_MAT(0) = Translation(offset) * node->local_transform.SIMD_MAT(0);
//
//        // Inherit clipping bounds from the graph node
//        node->material.clip_min = graph_node->material.clip_min;
//        node->material.clip_max = graph_node->material.clip_max;
//        node->draggable = false;
//        // Add it to the structural hierarchy
//        graph_node->add_child(node);
//    }
//
//    // Override the draw method so we can steal the camera's zoom factor right before rendering!
//    virtual void draw(id<MTLRenderCommandEncoder> cmdEncoder,
//                      id<MTLDevice> metalDevice,
//                      id<MTLRenderPipelineState> __strong (&predefinedStates)[7],
//                      Camera3D* cam,
//                      int& active_state) override {
//
//        // 1. Calculate camera distance (zoom factor)
//        float zoom_dist = simd_length(cam->target - cam->position);
//
//        // 2. Base thickness for our axes (you can tweak this!)
//        float base_thickness = 0.002f;
//        float dynamic_thickness = base_thickness * zoom_dist;
//
//        // 3. Dynamically scale the axes so they always appear the same size on screen!
//        X_axis->local_transform.SIMD_MAT(0) = p_scale(100.0f, dynamic_thickness, 1.0f);
//        Y_axis->local_transform.SIMD_MAT(0) = p_scale(dynamic_thickness, 100.0f, 1.0f);
//
//        // 4. Update the dynamic grid geometry (Aditya's logic)
//        grid.update_grid(cam->position, zoom_dist);
//
//        // Call the standard viewer draw loop
//        Viewer::draw(cmdEncoder, metalDevice, predefinedStates, cam, active_state);
//    }
//
//    virtual bool handle_event(const ViewerEvent& event) override {
//        // Allow gizmo manipulation to take priority if active
//        if (Viewer::handle_event(event)) return true;
//
//        // Ensure we only interact when the mouse is over the graph frame
//        HitResult hit = ray_hit_closest(event.ray_origin, event.ray_dir, {graph_frame});
//        if (!hit.hit) return false;
//
//        if (event.type == MouseEvent::Drag || event.type == MouseEvent::Scroll) {
//            // Find plane geometry in world space
//            simd_float4x4 world_mat = graph_frame->world_transform.SIMD_MAT(0);
//            simd_float3 normal = world_mat.columns[2].xyz; // Z axis of the frame
//            simd_float3 origin = world_mat.columns[3].xyz;
//
//            // Project rays onto the plane
//            simd_float3 p_prev = ray_plane_dir(event.prev_ray_origin, event.prev_ray_dir, normal, origin);
//            simd_float3 p_curr = ray_plane_dir(event.ray_origin, event.ray_dir, normal, origin);
//
//            simd_float3 delta_pos = p_curr - p_prev;
//
//            // Apply translation directly exactly like Viewer.cpp does for Gizmos
//            graph_node->local_transform.SIMD_MAT(0) = Translation(delta_pos) * graph_node->local_transform.SIMD_MAT(0);
//            graph_node->local_transform.tape->evaluated = false;
//            return true;
//        }
//        else if (event.type == MouseEvent::Zoom) {
//            simd_float4x4 world_mat = graph_frame->world_transform.SIMD_MAT(0);
//            simd_float3 normal = world_mat.columns[2].xyz;
//            simd_float3 origin = world_mat.columns[3].xyz;
//            simd_float3 p_curr = ray_plane_dir(event.ray_origin, event.ray_dir, normal, origin);
//
//            // Scale around the mouse cursor in world space natively
//            float scale_factor = 1.0f + event.zoom_magnitude;
//            graph_node->local_transform.SIMD_MAT(0) = Translation(p_curr) * Scale(simd_make_float3(scale_factor, scale_factor, 1.0f)) * Translation(-p_curr) * graph_node->local_transform.SIMD_MAT(0);
//            graph_node->local_transform.tape->evaluated = false;
//            return true;
//        }
//
//        return false;
//    }
//
//    void render_graph(const matrix& x, const matrix& y, float line_width = 0.0025f) {
//        matrix points = matrix::stack({x, y}, -1);
//
//        // Pad with z = 0
//        points = matrix::concat({points, matrix::zeros({points.shape()[0], 1})}, -1);
//
//        GeoNodeImpl node_to_update = graph_node;
//
//        dispatch_async(dispatch_get_main_queue(), ^{
//            node_to_update->mesh.vert_position = points;
//            node_to_update->mesh.indices = matrix(0, 0, dtype::UInt32); // Handled by vertex shader!
//            node_to_update->material.pipeline_state = 4;
//            node_to_update->material.line_width = line_width;
//        });
//    }
//};


// Loads a model file (.obj, .usdz, .usd, ...) through ModelIO. `node` is an empty root; the model hangs
// under it as one child per material (each child's Material holds one colour/texture), with positions and
// normals baked into the root's space. Tapping any part selects the root, so the model moves as one.
class ModelController {
public:
    GeoNodeImpl node; // Root node for the model

    // Pure C++ interface. We will handle the Objective-C++ NSURL conversion internally.
    ModelController(std::string name, std::string filepath, bool use_matrix_strides = false, matrix anchor = matrix(0, 0, dtype::Float)) {
        node = GeoNode::create(name);
        NSString* path = [NSString stringWithUTF8String:filepath.c_str()];
        NSURL* url = [NSURL fileURLWithPath:path];
        
        NSLog(@"url, %@", url);
        
        MDLAsset* asset = [[MDLAsset alloc] initWithURL:url];
        
        if (asset == nil || asset.count == 0) {
            printf("Failed to load 3D model at path: %s\n", filepath.c_str());
            return;
        }
        [asset loadTextures];   // material properties that name an image become MDLTextures
        
        MDLVertexDescriptor* model_vertex_desc = [[MDLVertexDescriptor alloc] init];
        if (use_matrix_strides) {
            // Option B (AoS): Pack everything tightly into ONE interleaved buffer (bufferIndex: 0)
            size_t offset = 0;
            model_vertex_desc.attributes[0] = [[MDLVertexAttribute alloc] initWithName:MDLVertexAttributePosition format:MDLVertexFormatFloat3 offset:offset bufferIndex:0];
            offset+= sizeof(float)*3;
            model_vertex_desc.attributes[1] = [[MDLVertexAttribute alloc] initWithName:MDLVertexAttributeTextureCoordinate format:MDLVertexFormatFloat2 offset:offset bufferIndex:0];
            offset+= sizeof(float)*2;
            model_vertex_desc.attributes[2] = [[MDLVertexAttribute alloc] initWithName:MDLVertexAttributeNormal format:MDLVertexFormatFloat3 offset:offset bufferIndex:0];
            offset+= sizeof(float)*3;
            model_vertex_desc.layouts[0] = [[MDLVertexBufferLayout alloc] initWithStride:offset];
        } else {
            // Option A (SoA): Force ModelIO to generate THREE separate contiguous buffers

            model_vertex_desc.attributes[0] = [[MDLVertexAttribute alloc] initWithName:MDLVertexAttributePosition format:MDLVertexFormatFloat3 offset:0 bufferIndex:0];

            model_vertex_desc.attributes[1] = [[MDLVertexAttribute alloc] initWithName:MDLVertexAttributeTextureCoordinate format:MDLVertexFormatFloat2 offset:0 bufferIndex:1];

            model_vertex_desc.attributes[2] = [[MDLVertexAttribute alloc] initWithName:MDLVertexAttributeNormal format:MDLVertexFormatFloat3 offset:0 bufferIndex:2];
       
            model_vertex_desc.layouts[0] = [[MDLVertexBufferLayout alloc] initWithStride:sizeof(float)*3]; // Pos
            model_vertex_desc.layouts[1] = [[MDLVertexBufferLayout alloc] initWithStride:sizeof(float)*2]; // UV
            model_vertex_desc.layouts[2] = [[MDLVertexBufferLayout alloc] initWithStride:sizeof(float)*3]; // Norm
            
        }
        
        // USD/USDZ (and most scene formats) nest meshes under transform nodes, so the asset's
        // top-level objects are usually plain MDLObjects. childObjectsOfClass: walks the whole
        // hierarchy; .obj files still come back as their top-level meshes.
        NSArray<MDLObject*>* meshes = [asset childObjectsOfClass:[MDLMesh class]];
        if (meshes.count == 0) {
            printf("No MDLMesh found in the asset!\n");
            return;
        }

        // One part per material (by name: the same material on many meshes is one part). Each part
        // gathers the vertices its triangles use, so parts don't carry each other's vertices.
        struct Part {
            MDLMaterial* material = nil;
            std::vector<float> positions, uvs, normals;
            std::vector<uint32_t> indices;
        };
        std::vector<Part> parts;
        std::unordered_map<std::string, size_t> part_of_material;

        for (MDLObject* obj in meshes) {
            MDLMesh* mesh = (MDLMesh*)obj;
            // Compute normals if the file doesn't have them (e.g. Stanford Bunny)
            [mesh addNormalsWithAttributeNamed:MDLVertexAttributeNormal creaseThreshold:0.85];
            [mesh setVertexDescriptor:model_vertex_desc];
            NSUInteger num_verts = mesh.vertexCount;
            if (num_verts == 0) continue;

            // Per attribute: base pointer and stride in floats
            const float *pos_data, *uv_data, *normal_data;
            size_t pos_stride, uv_stride, normal_stride;
            if (use_matrix_strides) {
                // Option B (AoS): one interleaved buffer, [pos(3) uv(2) normal(3)]
                const float* interleaved = (const float*)[mesh.vertexBuffers[0].map bytes];
                pos_data = interleaved;     pos_stride = 8;
                uv_data = interleaved + 3;  uv_stride = 8;
                normal_data = interleaved + 5; normal_stride = 8;
            } else {
                // Option A (SoA): ModelIO did the separation for us
                pos_data = (const float*)[mesh.vertexBuffers[0].map bytes];    pos_stride = 3;
                uv_data = (const float*)[mesh.vertexBuffers[1].map bytes];     uv_stride = 2;
                normal_data = (const float*)[mesh.vertexBuffers[2].map bytes]; normal_stride = 3;
            }

            simd_float4x4 world = [MDLTransform globalTransformWithObject:mesh atTime:0];
            simd_float3x3 world3 = simd_matrix(world.columns[0].xyz, world.columns[1].xyz, world.columns[2].xyz);
            simd_float3x3 normal_mat = simd_transpose(simd_inverse(world3));

            // This mesh's vertex -> index in a part, per part it feeds (-1: not copied there yet)
            std::unordered_map<size_t, std::vector<int64_t>> remaps;

            for (MDLSubmesh* submesh in mesh.submeshes) {
                if (submesh.geometryType != MDLGeometryTypeTriangles) continue;
                std::string key = submesh.material.name ? submesh.material.name.UTF8String : "";
                auto found = part_of_material.find(key);
                size_t p = found != part_of_material.end() ? found->second : parts.size();
                if (p == parts.size()) {
                    part_of_material[key] = p;
                    parts.push_back(Part{submesh.material});
                }
                Part& part = parts[p];
                std::vector<int64_t>& remap = remaps[p];
                if (remap.empty()) remap.assign(num_verts, -1);

                const void* index_data = [submesh.indexBuffer.map bytes];
                NSUInteger num_indices = submesh.indexCount;
                // ModelIO can output 8, 16 or 32-bit indices. The matrix needs 32-bit.
                for (NSUInteger i = 0; i < num_indices; i++) {
                    uint32_t v;
                    switch (submesh.indexType) {
                        case MDLIndexBitDepthUInt8:  v = ((const uint8_t*)index_data)[i]; break;
                        case MDLIndexBitDepthUInt16: v = ((const uint16_t*)index_data)[i]; break;
                        default:                     v = ((const uint32_t*)index_data)[i]; break;
                    }
                    if (remap[v] < 0) {
                        remap[v] = (int64_t)(part.positions.size() / 3);
                        const float* pos = pos_data + v * pos_stride;
                        const float* uv = uv_data + v * uv_stride;
                        const float* n = normal_data + v * normal_stride;
                        simd_float4 wp = simd_mul(world, simd_make_float4(pos[0], pos[1], pos[2], 1.0f));
                        simd_float3 wn = simd_normalize(simd_mul(normal_mat, simd_make_float3(n[0], n[1], n[2])));
                        part.positions.insert(part.positions.end(), {wp.x, wp.y, wp.z});
                        part.uvs.insert(part.uvs.end(), {uv[0], uv[1]});
                        part.normals.insert(part.normals.end(), {wn.x, wn.y, wn.z});
                    }
                    part.indices.push_back((uint32_t)remap[v]);
                }
            }
        }

        if (parts.empty()) {
            printf("The asset's meshes have no triangles!\n");
            return;
        }

        // Opaque parts first: the Viewer draws in tree order and blends, so see-through parts (glass)
        // must come after whatever they show.
        std::vector<GeoNodeImpl> opaque, transparent;
        size_t total_verts = 0, total_tris = 0;
        for (Part& part : parts) {
            size_m num_verts = (size_m)(part.positions.size() / 3);
            if (num_verts == 0) continue;
            total_verts += num_verts;
            total_tris += part.indices.size() / 3;

            NSString* material_name = part.material.name ?: @"default";
            GeoNodeImpl child = GeoNode::create(name + "/" + material_name.UTF8String);
            child->select_parent = true;

            matrix positions = matrix::zeros({num_verts, 3}, dtype::Float);
            positions.eval(); // zeros() is now lazy; force the buffer to materialize before writing into it directly
            memcpy(positions.buffer, part.positions.data(), part.positions.size() * sizeof(float));
            matrix uvs = matrix::zeros({num_verts, 2}, dtype::Float);
            uvs.eval();
            memcpy(uvs.buffer, part.uvs.data(), part.uvs.size() * sizeof(float));
            matrix normals = matrix::zeros({num_verts, 3}, dtype::Float);
            normals.eval();
            memcpy(normals.buffer, part.normals.data(), part.normals.size() * sizeof(float));
            matrix indices = matrix::zeros({(size_m)part.indices.size()}, dtype::UInt32);
            indices.eval();
            memcpy(indices.buffer, part.indices.data(), part.indices.size() * sizeof(uint32_t));

            child->mesh.vert_position = positions;
            child->mesh.uv_coords = uvs;
            child->mesh.normals = normals;
            child->mesh.indices = indices;
            // Explicitly build the metal buffer so it's ready for the Viewer
            if (part.indices.size() > 10) {
                child->mesh.indices.buildMetalBuffer();
            }

            bool is_opaque = apply_material(child->material, part.material);
            (is_opaque ? opaque : transparent).push_back(child);
        }
        for (auto& child : opaque) node->add_child(child);
        for (auto& child : transparent) node->add_child(child);
        printf("ModelController: %lu meshes, %zu parts, %zu vertices, %zu triangles\n",
               (unsigned long)meshes.count, opaque.size() + transparent.size(), total_verts, total_tris);
        
        if (anchor.total_size > 0) {
            // If the mesh vertices are bound to the root node (e.g. flattened), apply anchor here
            // Note: If we use child GeoNodes for submeshes, we might apply the anchor to the root's local_transform instead.
        }
    }

private:
    // ModelIO's physically based material -> ours: base colour (or texture), opacity, metallic, roughness.
    // Files without materials (.obj without .mtl) keep a light grey. Returns whether the part is opaque.
    static bool apply_material(Material& out, MDLMaterial* material) {
        out.shading = Shading::Standard;
        simd_float4 color = {0.8f, 0.8f, 0.8f, 1.0f};
        out.metallic = 0.0f;
        out.roughness = 0.5f;

        if (material) {
            MDLMaterialProperty* base = [material propertyWithSemantic:MDLMaterialSemanticBaseColor];
            if (base.type == MDLMaterialPropertyTypeTexture && base.textureSamplerValue.texture) {
                id<MTLTexture> texture = load_texture(base.textureSamplerValue.texture);
                if (texture) {
                    color = {1.0f, 1.0f, 1.0f, 1.0f};   // the shader multiplies the texture by this
                    out.texture = texture;
                    out.has_texture = true;
                    out.texture_repeat = true;
                }
            } else if (base) {
                simd_float3 rgb = property_color(base);
                color = {rgb.x, rgb.y, rgb.z, 1.0f};
            }

            MDLMaterialProperty* metallic = [material propertyWithSemantic:MDLMaterialSemanticMetallic];
            if (metallic.type == MDLMaterialPropertyTypeFloat) out.metallic = metallic.floatValue;
            MDLMaterialProperty* roughness = [material propertyWithSemantic:MDLMaterialSemanticRoughness];
            if (roughness.type == MDLMaterialPropertyTypeFloat) out.roughness = roughness.floatValue;
            MDLMaterialProperty* opacity = [material propertyWithSemantic:MDLMaterialSemanticOpacity];
            if (opacity.type == MDLMaterialPropertyTypeFloat) color.w = opacity.floatValue;
        }

        out.colors = matrix::of<float>({color.x, color.y, color.z, color.w}).reshape(1, 4);
        out.casts_shadows = color.w >= 1.0f;   // glass shouldn't black out what's behind it in the shadow map
        return color.w >= 1.0f;
    }

    static simd_float3 property_color(MDLMaterialProperty* p) {
        switch (p.type) {
            case MDLMaterialPropertyTypeFloat:  return simd_make_float3(p.floatValue);
            case MDLMaterialPropertyTypeFloat3: return p.float3Value;
            case MDLMaterialPropertyTypeFloat4: return p.float4Value.xyz;
            case MDLMaterialPropertyTypeColor: {
                CGColorRef c = p.color;
                const CGFloat* k = CGColorGetComponents(c);
                return CGColorGetNumberOfComponents(c) >= 3 ? simd_make_float3((float)k[0], (float)k[1], (float)k[2])
                                                            : simd_make_float3((float)k[0]);
            }
            default: return simd_make_float3(0.8f);
        }
    }

    // Colour textures are sRGB-encoded: loading them as sRGB lets the sampler return linear values,
    // like the material's plain colours. Mipmaps keep large textures from shimmering at a distance.
    static id<MTLTexture> load_texture(MDLTexture* texture) {
        MTKTextureLoader* loader = [[MTKTextureLoader alloc] initWithDevice:GlobalGPUManager.metalDevice];
        NSDictionary* options = @{
            MTKTextureLoaderOptionSRGB: @YES,
            MTKTextureLoaderOptionGenerateMipmaps: @YES,
            MTKTextureLoaderOptionOrigin: MTKTextureLoaderOriginBottomLeft,
            MTKTextureLoaderOptionTextureUsage: @(MTLTextureUsageShaderRead),
            MTKTextureLoaderOptionTextureStorageMode: @(MTLStorageModePrivate),
        };
        NSError* error = nil;
        id<MTLTexture> result = [loader newTextureWithMDLTexture:texture options:options error:&error];
        if (!result) NSLog(@"ModelController: texture %@ failed to load: %@", texture.name, error);
        return result;
    }
};

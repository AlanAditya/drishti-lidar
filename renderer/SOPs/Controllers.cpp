#pragma once
#include <string>
#include <vector>
#include <simd/simd.h>
#include "matrix.h"
#include "GeoNode.cpp"
#include "MeshPrimitives.cpp"
#import <ModelIO/ModelIO.h>

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

    // One label per anchor (world position of the label's bottom centre), all the same colours.
    void update(const std::vector<simd_float3>& anchors, const std::vector<std::string>& texts,
                simd_float4 text_rgba = {0.72f, 0.93f, 1.0f, 1.0f},
                simd_float4 plate_rgba = {0.02f, 0.08f, 0.14f, 0.55f}) {
        const Atlas& at = atlas();
        const size_t n = std::min(anchors.size(), texts.size());
        size_t count = 0;
        for (size_t i = 0; i < n; ++i) count += 1 + texts[i].size();
        if (count == 0) { clear(); return; }

        const float advance = at.glyph_aspect, pad = 0.2f, lift = 0.35f;
        matrix inst = matrix::withShape({(size_m)count, 4, 4}, dtype::Float);
        inst.begin_refcount();
        size_t k = 0;
        auto put = [&](simd_float3 a, float layer, simd_float4 rect, simd_float4 uv, simd_float4 rgba) {
            simd_float4x4& m = inst.SIMD_MAT((int)k++);
            m.columns[0] = simd_make_float4(a, layer);
            m.columns[1] = rect;
            m.columns[2] = uv;
            m.columns[3] = rgba;
        };
        for (size_t i = 0; i < n; ++i) {
            const std::string& s = texts[i];
            const float w = advance * s.size();
            const float x0 = -0.5f * w, y0 = lift;
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
            nd->material.line_width = sz;
            nd->set_local_transform(inst);
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

class TriangleController {
public:
    GeoNodeImpl node;

    TriangleController(std::string name, matrix anchor = matrix(0, dtype::Float)) {
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

    CylinderController(std::string name, float radius = 1.0f, float length = 1.0f, int subdivisions = 32, matrix anchor = matrix(0, dtype::Float)) {
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

    ConeController(std::string name, float radius = 1.0f, float length = 1.0f, int subdivisions = 32, matrix anchor = matrix(0, dtype::Float)) {
        node = GeoNode::create(std::move(name));
        node->mesh = MeshPrimitives::cone(radius, length, subdivisions);
        if (anchor.total_size > 0) {
            node->mesh.vert_position = node->mesh.vert_position + anchor;
        }
    }
    
    void update(simd_float3 dir, float r) {
        int n = node->mesh.vert_position.shape()[0] - 2;
        simd_float3 up = {0.0f, 1.0f, 0.0f};
        // Avoid singularity if dir is parallel to up
        if (std::abs(simd_normalize(dir).y) > 0.99f) {
            up = {1.0f, 0.0f, 0.0f};
        }
        simd_float3 right = simd_normalize(simd_cross(up, dir));
        up = -simd_normalize(simd_cross(right, dir));
        
        node->mesh.vert_position.at<float>(0, 0) = 0.0f;
        node->mesh.vert_position.at<float>(0, 1) = 0.0f;
        node->mesh.vert_position.at<float>(0, 2) = 0.0f;
        
        node->mesh.vert_position.at<float>(n + 1, 0) = dir.x;
        node->mesh.vert_position.at<float>(n + 1, 1) = dir.y;
        node->mesh.vert_position.at<float>(n + 1, 2) = dir.z;
        
        float theta = 0;
        float stride = 2 * M_PI / n;
        for (int i = 0; i < n; i++) {
            simd_float3 resultant = r * right * cos(theta) + r * up * sin(theta);
            node->mesh.vert_position.at<float>(1 + i, 0) = resultant.x;
            node->mesh.vert_position.at<float>(1 + i, 1) = resultant.y;
            node->mesh.vert_position.at<float>(1 + i, 2) = resultant.z;
            theta += stride;
        }
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

    LineController(std::string name = "Line3D", float width = 0.0025f) : line_width(width) {
        node = GeoNode::create(std::move(name));
        node->material.pipeline_state = 4;
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
        
        dispatch_async(dispatch_get_main_queue(), ^{
            node_to_update->mesh.vert_position = points; // Send raw points directly!
            node_to_update->mesh.indices = matrix(0, 0, dtype::UInt32); // Clear indices, handled by vertex shader triangle strip
            node_to_update->material.pipeline_state = 4; // Our new Vertex Shader path
            node_to_update->material.line_width = current_line_width;
        });
    }
    
    void update(matrix x, matrix y) {
        if (x.type != dtype::Float) x = x.astype(dtype::Float);
        if (y.type != dtype::Float) y = y.astype(dtype::Float);
        matrix points = matrix::stack({x, y, x.zeros()}, -1);
        
        GeoNodeImpl node_to_update = node;
        float current_line_width = line_width;
        
        dispatch_async(dispatch_get_main_queue(), ^{
            node_to_update->mesh.vert_position = points; // Send raw points directly!
            node_to_update->mesh.indices = matrix(0, 0, dtype::UInt32); // Clear indices, handled by vertex shader triangle strip
            node_to_update->material.pipeline_state = 4; // Our new Vertex Shader path
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


class ModelController {
public:
    GeoNodeImpl node; // Root node for the model

    // Pure C++ interface. We will handle the Objective-C++ NSURL conversion internally.
    ModelController(std::string name, std::string filepath, bool use_matrix_strides = false, matrix anchor = matrix(0, dtype::Float)) {
        node = GeoNode::create(std::move(name));
        NSString* path = [NSString stringWithUTF8String:filepath.c_str()];
        NSURL* url = [NSURL fileURLWithPath:path];
        
        NSLog(@"url, %@", url);
        
        MDLAsset* asset = [[MDLAsset alloc] initWithURL:url];
        
        if (asset == nil || asset.count == 0) {
            printf("Failed to load 3D model at path: %s\n", filepath.c_str());
            return;
        }
        
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
        
        MDLMesh* mesh = NULL;
        for (int i = 0; i < asset.count; i++) {
            MDLObject* obj = [asset objectAtIndex:i];
            if ([obj isKindOfClass:[MDLMesh class]]) {
                mesh = (MDLMesh*)obj;
                break;
            }
        }
        if (!mesh) {
            printf("No MDLMesh found in the asset!\n");
            return;
        }
        
        // Compute normals if the obj file doesn't have them (e.g. Stanford Bunny)
        [mesh addNormalsWithAttributeNamed:MDLVertexAttributeNormal creaseThreshold:0.85];
        [mesh setVertexDescriptor:model_vertex_desc];
        NSUInteger num_verts = mesh.vertexCount;
        
        matrix root_positions(0, dtype::Float);
        matrix root_uvs(0, dtype::Float);
        matrix root_normals(0, dtype::Float);
        
        if (use_matrix_strides) {
            id<MDLMeshBuffer> interleavedBuffer = mesh.vertexBuffers[0];
            void *interleavedData = [interleavedBuffer.map bytes];
            matrix raw_block = matrix::zeros({(size_m)num_verts, 8}, dtype::Float);
            raw_block.eval(); // zeros() is now lazy; force the buffer to materialize before writing into it directly
            memcpy(raw_block.buffer, interleavedData, num_verts * 8 * sizeof(float));
            root_positions = raw_block[R(), R(0, 3)].astype(dtype::Float, true);
            root_uvs = raw_block[R(), R(3, 5)].astype(dtype::Float, true);
        } else {
            // === OPTION A (SoA): Let ModelIO do the separation for us ===
            id<MDLMeshBuffer> posBuffer = mesh.vertexBuffers[0];
            id<MDLMeshBuffer> uvBuffer  = mesh.vertexBuffers[1];
            id<MDLMeshBuffer> normalBuffer  = mesh.vertexBuffers[2];
            
            void *posData = [posBuffer.map bytes];
            void *uvData  = [uvBuffer.map bytes];
            void *normalData  = [normalBuffer.map bytes];
            // Allocate matrix tensors and copy the separated contiguous memory in
            root_positions = matrix::zeros({(size_m)num_verts, 3}, dtype::Float);
            root_positions.eval(); // zeros() is now lazy; force the buffer to materialize before writing into it directly
            memcpy(root_positions.buffer, posData, num_verts * 3 * sizeof(float));

            root_uvs = matrix::zeros({(size_m)num_verts, 2}, dtype::Float);
            root_uvs.eval();
            memcpy(root_uvs.buffer, uvData, num_verts * 2 * sizeof(float));

            root_normals = matrix::zeros({(size_m)num_verts, 3}, dtype::Float);
            root_normals.eval();
            memcpy(root_normals.buffer, normalData, num_verts * 3 * sizeof(float));
        }
        
        // TODO: Implement ModelIO parsing here!
        node->mesh.vert_position = root_positions;
        node->mesh.uv_coords = root_uvs;
        if (mesh.submeshes.count > 0) {
            MDLSubmesh *firstSubmesh = mesh.submeshes.firstObject;
            id<MDLMeshBuffer> indexBuffer = firstSubmesh.indexBuffer;
            void *indexData = [indexBuffer.map bytes];
            NSUInteger num_indices = firstSubmesh.indexCount;
            
            matrix root_indices = matrix::zeros({(size_m)num_indices}, dtype::UInt32);
            root_indices.eval(); // zeros() is now lazy; force the buffer to materialize before writing into it directly

            // ModelIO can output 16-bit or 32-bit indices. Your matrix needs 32-bit.
            if (firstSubmesh.indexType == MDLIndexBitDepthUInt16) {
                uint16_t *indices16 = (uint16_t *)indexData;
                uint32_t *indices32 = (uint32_t *)root_indices.buffer;
                for (NSUInteger i = 0; i < num_indices; i++) {
                    indices32[i] = (uint32_t)indices16[i];
                }
            } else if (firstSubmesh.indexType == MDLIndexBitDepthUInt32) {
                memcpy(root_indices.buffer, indexData, num_indices * sizeof(uint32_t));
            }
            
            node->mesh.indices = root_indices;
            
            // Pad [N, 3] normals to [N, 4] so they match the float4 color expectation in the shader
            node->material.colors = root_normals.pad(0, 1, -1, matrix::scalar(1.0f));
            // Explicitly build the metal buffer so it's ready for the Viewer
            if (num_indices > 10) {
                node->mesh.indices.buildMetalBuffer();
            }
        }
        
        if (anchor.total_size > 0) {
            // If the mesh vertices are bound to the root node (e.g. flattened), apply anchor here
            // Note: If we use child GeoNodes for submeshes, we might apply the anchor to the root's local_transform instead.
        }
    }
};

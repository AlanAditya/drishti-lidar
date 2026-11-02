#pragma once
#include <vector>
#include <simd/simd.h>
#include "GeoNode.cpp"
#include "./../matrix/primitives.cpp"
#include "./../Mods/Utils.h"
#include "./../matrix/matrix.h"
#include "Controllers.cpp"

// Forward declarations to make code compile if necessary (conceptual)
// struct RenderBucketJob { Mesh mesh; Material material; matrix instance_buffer; uint32_t instance_count; };
// class GlobalRenderer { static void submit(const RenderBucketJob&); };

class Camera3D; // Forward declaration


inline bool HitTriangles(matrix triangles, simd_float3 ray_origin, simd_float3 ray_direction) {
    // triangles = [N, 3]
    
    triangles = triangles.reshape((triangles.shape()[0] / 3), (uint32_t)3, (uint32_t)3);
    
    // Finding Normals
    matrix v0 = triangles[R(), 0]; // [N/3, 3]
    matrix v1 = triangles[R(), 1]; // [N/3, 3]
    matrix v2 = triangles[R(), 2]; // [N/3, 3]
    
    matrix e0 = v1-v0; // [N/3, 3]
    matrix e1 = v2-v1; // [N/3, 3]
    matrix e2 = v0-v2; // [N/3, 3]
    
    matrix norms = matrix::cross(e1, e2); // [N/3, 3]
    
    // Finding PlanePoint
    // r.n = d
    // r = bt + c
    
    // (bt+c).n = d
    // t = (d - c.n) / (b.n)
    matrix ray_org = { ray_origin.x, ray_origin.y, ray_origin.z };
    matrix ray_dir = { ray_direction.x, ray_direction.y, ray_direction.z };
    
    matrix d = (v0 * norms).sum(-1, true); //  [N/3, 3] dot [N/3, 3] = [N/3, 1]
    matrix t = (d - (ray_org * norms).sum(-1, true) ) / (ray_dir * norms).sum(-1, true); // [N/3, 1]
    matrix p = t * ray_dir + ray_org; // [N/3, 3]
    
    // finding if intersects
    matrix m0 = p-v0; // [N/3, 3]
    matrix m1 = p-v1; // [N/3, 3]
    matrix m2 = p-v2; // [N/3, 3]
    
    matrix c0 = matrix::cross(e0, m0);
    matrix c1 = matrix::cross(e1, m1);
    matrix c2 = matrix::cross(e2, m2);
    
    matrix d0 = (c0 * norms).sum(-1);
    matrix d1 = (c1 * norms).sum(-1);
    matrix d2 = (c2 * norms).sum(-1);
    
    float max0 = d0.max().at<float>();
    float max1 = d1.max().at<float>();
    float max2 = d2.max().at<float>();
    
    float min0 = d0.min().at<float>();
    float min1 = d1.min().at<float>();
    float min2 = d2.min().at<float>();
    
    return (min0 >= 0 && min1 >= 0 && min2 >= 0) || (max0 <= 0 && max1 <= 0 && max2 <= 0);
}

inline bool point_inside_triangle(simd_float3 p, simd_float3 p1, simd_float3 p2, simd_float3 p3) {
    simd_float3 e1 = p2-p1;
    simd_float3 e2 = p3-p2;
    simd_float3 e3 = p1-p3;
    
    simd_float3 m1 = p-p1;
    simd_float3 m2 = p-p2;
    simd_float3 m3 = p-p3;
    
    simd_float3 c1 = simd_cross(e1, m1);
    simd_float3 c2 = simd_cross(e2, m2);
    simd_float3 c3 = simd_cross(e3, m3);
    
    simd_float3 n = simd_cross(e1, e2); // triangle's normal, for reference orientation

    float d1 = simd_dot(c1, n);
    float d2 = simd_dot(c2, n);
    float d3 = simd_dot(c3, n);

    // Add a tiny epsilon to account for floating point noise when ray hits near triangle edges
    float eps = -1e-6f;

    return (d1 >= 0 && d2 >= 0 && d3 >= 0) || (d1 <= 0 && d2 <= 0 && d3 <= 0);
}

inline simd_float3 plane_norm(simd_float3 p1, simd_float3 p2, simd_float3 p3) {
    return simd_cross(p2-p1, p3-p1);
}

inline simd_float3 ray_plane_dir(simd_float3 ray_origin, simd_float3 ray_dir, simd_float3 plane_norm, simd_float3 plane_point) {
    // r.n = d
    // r = bt + c
    
    // (bt+c).n = d
    // t = (d - c.n) / (b.n)
    float d = simd_dot(plane_point, plane_norm);
    float t = ( d - simd_dot(ray_origin, plane_norm) )/ simd_dot(ray_dir, plane_norm);
    return t * ray_dir + ray_origin;
}


// View-projection for a directional light: an orthographic box looking along `dir`, centred on `centre`,
// reaching `radius` in every direction. Same conventions as Camera3D: column vectors (M * v), view-space
// +z forward, x = cross(up, z), depth 0..1.
inline simd_float4x4 light_view_proj(simd_float3 dir, simd_float3 centre, float radius) {
    simd_float3 z = simd_normalize(dir);                                   // forward: the way the light travels
    simd_float3 up = std::fabs(z.y) > 0.99f ? simd_make_float3(1, 0, 0) : simd_make_float3(0, 1, 0);
    simd_float3 x = simd_normalize(simd_cross(up, z));
    simd_float3 y = simd_cross(z, x);
    simd_float3 eye = centre - z * radius;                                 // the box starts `radius` before the centre
    simd_float4x4 view = simd_matrix_from_rows(simd_make_float4(x, -simd_dot(x, eye)),
                                               simd_make_float4(y, -simd_dot(y, eye)),
                                               simd_make_float4(z, -simd_dot(z, eye)),
                                               simd_make_float4(0, 0, 0, 1));
    simd_float4x4 proj = simd_matrix_from_rows(simd_make_float4(1 / radius, 0, 0, 0),        // x, y: [-r, r] -> [-1, 1]
                                               simd_make_float4(0, 1 / radius, 0, 0),
                                               simd_make_float4(0, 0, 1 / (2 * radius), 0),  // z: [0, 2r] -> [0, 1]
                                               simd_make_float4(0, 0, 0, 1));
    return simd_mul(proj, view);
}

enum class DragAxis {
    None, X, Y, Z, XY, YZ, ZX, Free
};

struct HitResult {
    bool hit;
    float distance;
    GeoNodeImpl node;
    simd_float3 hit_point;
};

class Viewer {
public:
    uint64_t current_pass_id = 0;
    std::vector<GeoNodeImpl> nodes; // The nodes managed by this viewer
    
    GeoNodeImpl active_node = nullptr;
    GizmoController transform_gizmo;
    DragAxis current_drag_axis = DragAxis::None;

    // Where and how the renderer draws this viewer. The defaults are the old behaviour: always,
    // over the whole drawable, with the renderer's camera.
    bool visible = true;                   // false: not drawn, and gets no events
    simd_float4 viewport = {0, 0, 1, 1};   // x, y, width, height as fractions of the drawable; origin top left
    // Null = the renderer's camera (its aspect ratio is the whole view's, so give a viewer that
    // doesn't cover the view its own). The renderer keeps this one's aspect ratio in step with the viewport.
    std::shared_ptr<Camera3D> camera;

    // Pages only (Renderer::pages, iOS): whether orbit / pan / zoom move this page's camera, and
    // whether the infinite grid is drawn behind it.
    bool interactive = true;
    bool show_grid = true;

    // Every matrix draw() bound to the GPU this frame. Metal reads their host memory directly
    // (buildMetalBuffer: newBufferWithBytesNoCopy, no deallocator) after draw() returns, so the
    // renderer takes these and keeps them alive until the frame's command buffer completes.
    // Otherwise a controller swapping in new data frees the old buffer while the GPU still reads it.
    std::vector<matrix> frame_refs;

    // Lighting for Shading::Lit meshes. The default is a key light from above-left in front of the
    // default camera, a dim fill from the right, and some ambient. Up to kMaxLights are used.
    static constexpr size_t kMaxLights = 8;
    std::vector<Light> lights = {
        Light::directional(simd_make_float3(0.5f, -0.8f, 0.6f), 0.85f),    // key
        Light::directional(simd_make_float3(-0.7f, -0.2f, 0.4f), 0.25f),   // fill
    };
    simd_float3 ambient = {0.15f, 0.15f, 0.15f};
    ToneMapping tone_mapping = ToneMapping::None;   // opt-in, like three.js; existing lit scenes keep their look
    float exposure = 1.0f;                          // multiplies lit colour before tone mapping
    Fog fog;                                        // off by default: existing scenes don't change

    // --- shadows (see prepare) ---
    uint32_t shadow_map_size = 2048;
    id<MTLTexture> shadow_map = nil;                 // depth as the shadow-casting light sees the casters
    simd_float4x4 shadow_light_vp = matrix_identity_float4x4;
    int32_t shadow_light = -1;                       // index in `lights` of the light with a shadow map; -1 none
    id<MTLTexture> shadow_dummy_ = nil;              // 1x1, bound when there's no shadow map (the shader declares one)
    id<MTLDepthStencilState> shadow_depth_state_ = nil;

    void ensure_shadow_dummy(id<MTLDevice> device) {
        if (shadow_dummy_) return;
        MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                                                     width:1 height:1 mipmapped:NO];
        d.usage = MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget;
        d.storageMode = MTLStorageModePrivate;
        shadow_dummy_ = [device newTextureWithDescriptor:d];
    }

    // Renders the shadow map, before the main pass: the host calls it on the frame's command buffer before
    // it opens its render pass. Only the first directional light with casts_shadows gets one, covering
    // the bounding box of every visible mesh (pipeline 3) with material.casts_shadows.
    virtual void prepare(id<MTLCommandBuffer> cmd, id<MTLDevice> device, id<MTLRenderPipelineState> shadow_state) {
        ensure_shadow_dummy(device);
        shadow_light = -1;
        for (size_t i = 0; i < std::min(lights.size(), kMaxLights); i++) {
            if (lights[i].casts_shadows && lights[i].type == Light::Type::Directional) { shadow_light = (int32_t)i; break; }
        }
        if (shadow_light < 0 || !shadow_state) return;

        std::vector<GeoNodeImpl> casters;
        for (const auto& root : nodes)
            for (const auto& n : gather_nodes(root))
                if (n->visible && !n->mesh.is_empty() && n->material.pipeline_state == 3 && n->material.casts_shadows)
                    casters.push_back(n);
        if (casters.empty()) { shadow_light = -1; return; }

        // World-space box around every caster instance: its translation +- its largest scale * the mesh's extent.
        simd_float3 lo = simd_make_float3(INFINITY, INFINITY, INFINITY), hi = -lo;
        for (auto& n : casters) {
            n->mesh.vert_position.eval();
            n->world_transform.eval();
            n->mesh.indices.eval();
            for (matrix* m : {&n->mesh.vert_position, &n->world_transform, &n->mesh.indices}) {
                if (m->total_size > 0 && !m->metalBuffer) m->buildMetalBuffer();
                if (!m->refCount && !m->tape && m->buffer && !(m->flags & NON_OWNERSHIP_FLAG)) m->begin_refcount();
                frame_refs.push_back(*m);
            }
            const matrix& p = n->mesh.vert_position;
            const size_m stride = p.dims == 2 ? p.shape()[1] : 3, count = (size_m)(p.total_size / stride);
            const float* v = static_cast<const float*>(p.buffer);
            float extent = 0.0f;
            for (size_m k = 0; k < count; k++)
                extent = std::max(extent, simd_length(simd_make_float3(v[k * stride], v[k * stride + 1], v[k * stride + 2])));
            for (size_m i = 0; i < n->world_transform.shape()[0]; i++) {
                simd_float4x4 w = n->world_transform.SIMD_MAT((int)i);
                float s = std::max({simd_length(w.columns[0].xyz), simd_length(w.columns[1].xyz), simd_length(w.columns[2].xyz)});
                lo = simd_min(lo, w.columns[3].xyz - s * extent);
                hi = simd_max(hi, w.columns[3].xyz + s * extent);
            }
        }
        const simd_float3 centre = (lo + hi) * 0.5f;
        const float radius = std::max(simd_length(hi - lo) * 0.5f, 1e-3f) * 1.05f;
        shadow_light_vp = light_view_proj(lights[shadow_light].direction, centre, radius);

        if (!shadow_map || shadow_map.width != shadow_map_size) {
            MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatDepth32Float
                                                                                         width:shadow_map_size height:shadow_map_size mipmapped:NO];
            d.usage = MTLTextureUsageRenderTarget | MTLTextureUsageShaderRead;
            d.storageMode = MTLStorageModePrivate;
            shadow_map = [device newTextureWithDescriptor:d];
        }
        if (!shadow_depth_state_) {
            MTLDepthStencilDescriptor* ds = [MTLDepthStencilDescriptor new];
            ds.depthCompareFunction = MTLCompareFunctionLess;
            ds.depthWriteEnabled = YES;
            shadow_depth_state_ = [device newDepthStencilStateWithDescriptor:ds];
        }

        MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
        pass.depthAttachment.texture = shadow_map;
        pass.depthAttachment.loadAction = MTLLoadActionClear;
        pass.depthAttachment.clearDepth = 1.0;
        pass.depthAttachment.storeAction = MTLStoreActionStore;
        id<MTLRenderCommandEncoder> enc = [cmd renderCommandEncoderWithDescriptor:pass];
        [enc setRenderPipelineState:shadow_state];
        [enc setDepthStencilState:shadow_depth_state_];
        [enc setCullMode:MTLCullModeNone];
        [enc setDepthBias:2.0f slopeScale:2.0f clamp:0.01f];      // pushes stored depth back a little: less self-shadowing
        [enc setVertexBytes:&shadow_light_vp length:sizeof(simd_float4x4) atIndex:2];
        for (auto& n : casters) {
            [enc setVertexBuffer:(id<MTLBuffer>)n->mesh.vert_position.metalBuffer offset:0 atIndex:0];
            [enc setVertexBuffer:(id<MTLBuffer>)n->world_transform.metalBuffer offset:0 atIndex:1];
            [enc drawIndexedPrimitives:MTLPrimitiveTypeTriangle indexCount:n->mesh.indices.total_size
                             indexType:MTLIndexTypeUInt32 indexBuffer:(id<MTLBuffer>)n->mesh.indices.metalBuffer
                     indexBufferOffset:0 instanceCount:n->world_transform.shape()[0]];
        }
        [enc endEncoding];
    }

    virtual ~Viewer() = default;

    // --- Drawing into this viewer. Each call adds a node and returns the controller that owns it:
    // keep the controller and call update() / clear() on it (safe from any thread: the swap lands
    // on the main queue, between frames), or reach the node itself through `.node`.

    // points [N, 3] Float; colors [N, 4] or [4] Float RGBA.
    PointCloudController points(matrix pts, matrix colors, float point_size = 5.0f) {
        PointCloudController cloud("points");
        cloud.point_size = point_size;
        nodes.push_back(cloud.node);
        cloud.update(pts, colors);
        return cloud;
    }

    // vertices [V, 3] Float, colours [V, 4] Float RGBA, indices [3T] UInt32.
    TriangleMeshController mesh(matrix vertices, matrix colours, matrix indices) {
        TriangleMeshController m("mesh");
        nodes.push_back(m.node);
        m.update(vertices, colours, indices);
        return m;
    }

    // img [H, W, 3|4], UInt8 or Float in [0, 1]: a quad 1 tall and W/H wide, row 0 at the top.
    // Leave img out to fill it later with update(). Also turns the ground grid off and, if this
    // viewer has its own camera, frames the image head-on, its height filling the view (zoom out
    // for a wide image in a narrow tile).
    ImageController image(matrix img = matrix(0, 0, dtype::UInt8), ImageController::Order order = ImageController::Order::RGBA) {
        ImageController pic("image");
        nodes.push_back(pic.node);
        if (img.dims != 0) pic.update(img, order);
        show_grid = false;
        if (camera) camera->lookAt(simd_make_float3(0, 0, -0.52f / std::tan((float)M_PI * 0.125f)), simd_make_float3(0, 0, 0));
        return pic;
    }

    // Any controller with a `node` (WireBoxController, LabelController, ModelController, ...).
    template <class Controller>
    Controller add(Controller controller) {
        nodes.push_back(controller.node);
        return controller;
    }

    virtual void draw(id<MTLRenderCommandEncoder> cmdEncoder, 
                      id<MTLDevice> metalDevice, 
                      id<MTLRenderPipelineState> __strong (&predefinedStates)[9], 
                      Camera3D* cam, 
                      int& active_state) {
        current_pass_id +=1;

        // This frame's lights, in the layout DAGNodeFragmentShader reads (one entry minimum: Metal wants the buffer).
        GPULight gpu_lights[kMaxLights] = {};
        uint32_t light_count = (uint32_t)std::min(lights.size(), kMaxLights);
        for (uint32_t i = 0; i < light_count; i++) {
            const Light& l = lights[i];
            simd_float3 dir = simd_length(l.direction) > 0 ? simd_normalize(l.direction) : simd_make_float3(0, -1, 0);
            gpu_lights[i].position_type = simd_make_float4(l.position, (float)(uint32_t)l.type);
            gpu_lights[i].direction_range = simd_make_float4(dir, l.range);
            gpu_lights[i].color_intensity = simd_make_float4(l.color, l.intensity);
        }
        
        if (active_node) {
//            transform_gizmo.node->local_transform.flags |= NON_OWNERSHIP_FLAG;
//            transform_gizmo.node->local_transform = active_node->world_transform;
            simd_float3 centre = active_node->world_transform.SIMD_MAT(0).columns[3].xyz;
            float zoom_dist = simd_length(centre - cam->position);
            transform_gizmo.node->local_transform.SIMD_MAT(0) = active_node->world_transform.SIMD_MAT(0) * Scale(0.1 * zoom_dist);
            transform_gizmo.node->local_transform.tape->version = ++global_epoch;
        }
        
        std::vector<GeoNodeImpl> scene_nodes;
        for (auto& node : nodes) {
            auto gathered = gather_nodes(node);
            scene_nodes.insert(scene_nodes.end(), gathered.begin(), gathered.end());
        }
        
        std::vector<GeoNodeImpl> gizmo_nodes;
        if (active_node) {
            gizmo_nodes = gather_nodes(transform_gizmo.node);
        }

        std::vector<GeoNodeImpl> mesh_nodes, pc_nodes, line_nodes, edge_nodes, label_nodes;
        split_nodes(scene_nodes, mesh_nodes, pc_nodes, line_nodes, edge_nodes, label_nodes);

        std::vector<GeoNodeImpl> gizmo_mesh_nodes, gizmo_pc_nodes, gizmo_line_nodes, gizmo_edge_nodes;
        std::vector<GeoNodeImpl> gizmo_label_nodes;
        split_nodes(gizmo_nodes, gizmo_mesh_nodes, gizmo_pc_nodes, gizmo_line_nodes, gizmo_edge_nodes, gizmo_label_nodes);
        
        DepthBias current_bias = {false, 0.0f, 0.0f, 0.0f};
        auto apply_bias = [&](const DepthBias& target) {
            if (current_bias.custom != target.custom || 
                current_bias.bias != target.bias || 
                current_bias.slopeScale != target.slopeScale || 
                current_bias.clamp != target.clamp) {
                
                [cmdEncoder setDepthBias:target.bias slopeScale:target.slopeScale clamp:target.clamp];
                current_bias = target;
            }
        };

        auto invalidate_nodes = [&](const std::vector<GeoNodeImpl>& list_nodes) {
            for (const auto& node : list_nodes) {
                if (node->mesh.is_empty()) continue;
                if (node->mesh.vert_position.tape) node->mesh.vert_position.tape->invalidate_pass(current_pass_id);
                if (node->world_transform.tape) node->world_transform.tape->invalidate_pass(current_pass_id);
                if (node->material.colors.tape) node->material.colors.tape->invalidate_pass(current_pass_id);
                if (node->mesh.uv_coords.tape) node->mesh.uv_coords.tape->invalidate_pass(current_pass_id);
                if (node->mesh.indices.tape) node->mesh.indices.tape->invalidate_pass(current_pass_id);
                if (node->mesh.normals.tape) node->mesh.normals.tape->invalidate_pass(current_pass_id);
                if (node->mesh.glyphs.tape) node->mesh.glyphs.tape->invalidate_pass(current_pass_id);
            }
        };
        invalidate_nodes(scene_nodes);
        invalidate_nodes(line_nodes);
        invalidate_nodes(edge_nodes);
        invalidate_nodes(label_nodes);
        invalidate_nodes(gizmo_nodes);
        invalidate_nodes(gizmo_line_nodes);
        invalidate_nodes(gizmo_edge_nodes);

        auto draw_nodes = [&](const std::vector<GeoNodeImpl>& render_nodes, const simd_float4x4& vMatrix) {
            for (const auto& node : render_nodes) {
                if (node->mesh.is_empty()) continue;

                node->mesh.vert_position.eval();
                node->world_transform.eval();
                node->local_transform.eval();
                node->material.colors.eval();
                node->mesh.uv_coords.eval();
                node->mesh.indices.eval();
                node->mesh.normals.eval();
                node->mesh.glyphs.eval();
                node->topology.edges.eval();

                auto ensure_metal = [](matrix& mat) {
                    if (mat.total_size > 0 && mat.metalBuffer == nullptr) {
                        mat.buildMetalBuffer();
                    }
                };
                
                ensure_metal(node->mesh.vert_position);                ensure_metal(node->local_transform);
                ensure_metal(node->world_transform);
                ensure_metal(node->material.colors);
                ensure_metal(node->mesh.uv_coords);
                ensure_metal(node->mesh.indices);
                ensure_metal(node->mesh.normals);
                ensure_metal(node->mesh.glyphs);
                ensure_metal(node->topology.edges);

                for (matrix* m : {&node->mesh.vert_position, &node->local_transform, &node->world_transform,
                                  &node->material.colors, &node->mesh.uv_coords, &node->mesh.indices,
                                  &node->mesh.normals, &node->mesh.glyphs, &node->topology.edges}) {
                    if (m->total_size == 0) continue;
                    // A plain owned buffer (no refcount, no tape, e.g. matrix{r, g, b, a}) would be deep-copied
                    // here, keeping a copy alive instead of the buffer the GPU reads. Make it refcounted first
                    // (as ensure_graph_ready does for tapeless inputs) so the copy shares it.
                    if (!m->refCount && !m->tape && m->buffer && !(m->flags & NON_OWNERSHIP_FLAG)) m->begin_refcount();
                    frame_refs.push_back(*m);
                }

                uint32_t target_state = node->material.pipeline_state;
                if (active_state != target_state) {
                    [cmdEncoder setRenderPipelineState:predefinedStates[target_state]];
                    active_state = target_state;
                }

                if (node->mesh.vert_position.metalBuffer) {
                    [cmdEncoder setVertexBuffer:(id<MTLBuffer>)node->mesh.vert_position.metalBuffer offset:0 atIndex:0];
                }
                if (node->world_transform.metalBuffer) {
                    [cmdEncoder setVertexBuffer:(id<MTLBuffer>)node->world_transform.metalBuffer offset:0 atIndex:1];
                }
                
                [cmdEncoder setVertexBytes:&vMatrix length:sizeof(simd_float4x4) atIndex:2];
                
                size_m instanceCount = node->world_transform.shape()[0];
                size_m vertexCount = node->mesh.vert_position.dims == 2 ? node->mesh.vert_position.shape()[0] : node->mesh.vert_position.total_size / 3;
                matrix b_colors = node->material.colors.broadcast_toV2({instanceCount, vertexCount, 4});
                
                struct { uint32_t inst; uint32_t vert; } color_strides = { 
                    (uint32_t)(b_colors.strides()[0] / 4), 
                    (uint32_t)(b_colors.strides()[1] / 4) 
                };
                
                [cmdEncoder setVertexBuffer:(id<MTLBuffer>)node->material.colors.metalBuffer offset:0 atIndex:3];
                [cmdEncoder setVertexBytes:&color_strides length:sizeof(color_strides) atIndex:4];
                
                if (node->mesh.uv_coords.metalBuffer) {
                    [cmdEncoder setVertexBuffer:(id<MTLBuffer>)node->mesh.uv_coords.metalBuffer offset:0 atIndex:5];
                }
                if (node->local_transform.metalBuffer) {
                    [cmdEncoder setVertexBuffer:(id<MTLBuffer>)node->local_transform.metalBuffer offset:0 atIndex:9];
                    uint32_t local_instances = node->local_transform.shape()[0];
                    [cmdEncoder setVertexBytes:&local_instances length:sizeof(uint32_t) atIndex:10];
                }

                if (active_state == 3) {   // mesh: normals (vertex 6, 7) and shading (fragment 3, 4)
                    // No normals: bind the positions in their place (the shader must have a buffer there) and
                    // say so; lit shading then uses flat face normals.
                    uint32_t has_normals = node->mesh.normals.metalBuffer && node->mesh.normals.total_size == node->mesh.vert_position.total_size;
                    id<MTLBuffer> normal_buffer = has_normals ? (id<MTLBuffer>)node->mesh.normals.metalBuffer
                                                              : (id<MTLBuffer>)node->mesh.vert_position.metalBuffer;
                    [cmdEncoder setVertexBuffer:normal_buffer offset:0 atIndex:6];
                    [cmdEncoder setVertexBytes:&has_normals length:sizeof(uint32_t) atIndex:7];
                    uint32_t shading = (uint32_t)node->material.shading;
                    [cmdEncoder setFragmentBytes:&shading length:sizeof(uint32_t) atIndex:3];
                    [cmdEncoder setFragmentBytes:&cam->position length:sizeof(simd_float3) atIndex:4];
                    [cmdEncoder setFragmentBytes:gpu_lights length:sizeof(GPULight) * std::max(light_count, 1u) atIndex:5];
                    [cmdEncoder setFragmentBytes:&light_count length:sizeof(uint32_t) atIndex:6];
                    [cmdEncoder setFragmentBytes:&ambient length:sizeof(simd_float3) atIndex:7];
                    simd_float2 surface = {node->material.specular, node->material.shininess};
                    [cmdEncoder setFragmentBytes:&surface length:sizeof(simd_float2) atIndex:8];
                    ensure_shadow_dummy(metalDevice);   // hosts that never call prepare() still bind something
                    [cmdEncoder setFragmentTexture:(shadow_light >= 0 ? shadow_map : shadow_dummy_) atIndex:1];
                    [cmdEncoder setFragmentBytes:&shadow_light_vp length:sizeof(simd_float4x4) atIndex:9];
                    int32_t node_shadow_light = node->material.receives_shadows ? shadow_light : -1;
                    [cmdEncoder setFragmentBytes:&node_shadow_light length:sizeof(int32_t) atIndex:10];
                    simd_float2 tone = {(float)(uint32_t)(node->material.tone_mapped ? tone_mapping : ToneMapping::None), exposure};
                    [cmdEncoder setFragmentBytes:&tone length:sizeof(simd_float2) atIndex:11];
                    simd_float2 pbr = {node->material.metallic, node->material.roughness};
                    [cmdEncoder setFragmentBytes:&pbr length:sizeof(simd_float2) atIndex:12];
                    Fog f = node->material.fog ? fog : Fog{};
                    simd_float4 fog_params = {(float)(uint32_t)f.mode, f.near, f.far, f.density};
                    [cmdEncoder setFragmentBytes:&fog_params length:sizeof(simd_float4) atIndex:13];
                    [cmdEncoder setFragmentBytes:&f.color length:sizeof(simd_float3) atIndex:14];
                    uint32_t texture_repeat = node->material.texture_repeat;
                    [cmdEncoder setFragmentBytes:&texture_repeat length:sizeof(uint32_t) atIndex:15];
                }

                if (active_state == 4 || active_state == 8) {   // thick lines: strip / round joins
                    [cmdEncoder setVertexBytes:&cam->position length:sizeof(simd_float3) atIndex:6];
                    [cmdEncoder setVertexBytes:&node->material.line_width length:sizeof(float) atIndex:7];
                    uint32_t total_points = node->mesh.vert_position.shape()[0];
                    [cmdEncoder setVertexBytes:&total_points length:sizeof(uint32_t) atIndex:8];
                }
                if (active_state == 5) {
                    [cmdEncoder setVertexBytes:&node->material.line_width length:sizeof(float) atIndex:7];
                }
                
                if (active_state == 6) {
                    if (node->topology.edges.metalBuffer) {
                        [cmdEncoder setVertexBuffer:(id<MTLBuffer>)node->topology.edges.metalBuffer offset:0 atIndex:8];
                    }
                    [cmdEncoder setVertexBytes:&cam->position length:sizeof(simd_float3) atIndex:6];
                    [cmdEncoder setVertexBytes:&node->material.line_width length:sizeof(float) atIndex:7];
                }

                if (active_state == 7) {   // text labels: the glyph records, drawn once per instance of the node
                    if (node->mesh.glyphs.metalBuffer) {
                        [cmdEncoder setVertexBuffer:(id<MTLBuffer>)node->mesh.glyphs.metalBuffer offset:0 atIndex:11];
                    }
                    // Glyph height (fraction of the viewport), aspect, glyphs per instance (LabelParams).
                    struct { float size; float aspect; uint32_t glyph_count; } label_params = {
                        node->material.line_width, cam->getAspectRatio(), (uint32_t)node->mesh.glyphs.shape()[0]};
                    [cmdEncoder setVertexBytes:&label_params length:sizeof(label_params) atIndex:7];
                    instanceCount *= label_params.glyph_count;
                }

                bool isTextured = node->material.has_texture;
                [cmdEncoder setFragmentBytes:&isTextured length:sizeof(bool) atIndex:0];
                if (isTextured && node->material.texture) {
                    [cmdEncoder setFragmentTexture:node->material.texture atIndex:0];
                }
                
                [cmdEncoder setFragmentBytes:&node->material.clip_min length:sizeof(simd_float3) atIndex:1];
                [cmdEncoder setFragmentBytes:&node->material.clip_max length:sizeof(simd_float3) atIndex:2];
                
                uint32_t indexCount = node->mesh.indices.total_size;
                
                if (active_state == 5) { // SOA Point Cloud
                    apply_bias(node->material.depth_bias);
                    uint32_t pointCount = node->mesh.vert_position.shape()[0];
                    [cmdEncoder drawPrimitives: MTLPrimitiveTypePoint 
                                   vertexStart: 0 
                                   vertexCount: pointCount 
                                 instanceCount: instanceCount];
                } else if (active_state == 6) { // SOA Edge Topology
                    apply_bias(node->material.depth_bias);
                    uint32_t num_edges = node->topology.edges.shape()[0];
                    [cmdEncoder drawPrimitives: MTLPrimitiveTypeTriangle
                                   vertexStart: 0 
                                   vertexCount: num_edges * 6
                                 instanceCount: instanceCount];
                } else if (active_state == 4) { // SOA Thick Line: one strip, 2 vertices per point
                    apply_bias(node->material.depth_bias);
                    uint32_t pointCount = node->mesh.vert_position.shape()[0];
                    [cmdEncoder drawPrimitives: MTLPrimitiveTypeTriangleStrip
                                   vertexStart: 0 
                                   vertexCount: pointCount * 2
                                 instanceCount: instanceCount];
                } else if (active_state == 8) { // Thick line with round joins: a quad per segment, then a disc per point (vertex_line3d_round)
                    apply_bias(node->material.depth_bias);
                    uint32_t pointCount = node->mesh.vert_position.shape()[0];
                    [cmdEncoder drawPrimitives: MTLPrimitiveTypeTriangle
                                   vertexStart: 0 
                                   vertexCount: pointCount * 12
                                 instanceCount: instanceCount];
                } else if (node->mesh.indices.metalBuffer) {
                    apply_bias(node->material.depth_bias);
                    
                    [cmdEncoder drawIndexedPrimitives: MTLPrimitiveTypeTriangle
                                           indexCount: indexCount 
                                            indexType: MTLIndexTypeUInt32 
                                          indexBuffer: node->mesh.indices.metalBuffer
                                    indexBufferOffset: 0
                                        instanceCount: instanceCount];
                }
            }
        };

        draw_nodes(mesh_nodes, cam->viewMatrix);
        draw_nodes(pc_nodes, cam->viewMatrix);
        draw_nodes(line_nodes, cam->viewMatrix);
        draw_nodes(edge_nodes, cam->viewMatrix);
        draw_nodes(label_nodes, cam->viewMatrix);   // last: translucent plates blend over the scene

        if (active_node) {
            simd_float4x4 overlayMatrix = cam->viewMatrix;
            // Compress the Z-depth by 99% to pull it entirely to the front clipping plane,
            // while preserving relative Z coordinates so gizmo components correctly depth-sort themselves.
            overlayMatrix.columns[0][2] *= 0.01f;
            overlayMatrix.columns[1][2] *= 0.01f;
            overlayMatrix.columns[2][2] *= 0.01f;
            overlayMatrix.columns[3][2] *= 0.01f;
            
            draw_nodes(gizmo_mesh_nodes, overlayMatrix);
            draw_nodes(gizmo_pc_nodes, overlayMatrix);
            draw_nodes(gizmo_line_nodes, overlayMatrix);
            draw_nodes(gizmo_edge_nodes, overlayMatrix);
        }
        
        DepthBias default_bias = {false, 0.0f, 0.0f, 0.0f};
        apply_bias(default_bias);
    }

    void render_scene(const GeoNodeImpl& root) {
        current_pass_id++;
        std::vector<GeoNodeImpl> flat_nodes = gather_nodes(root);
        for (const auto& node : flat_nodes) {
            if (node->mesh.is_empty()) continue;
        }
    }

private:
    std::vector<GeoNodeImpl> gather_nodes(const GeoNodeImpl& root) {
        std::vector<GeoNodeImpl> result;
        result.push_back(root);
        for (const auto& child : root->children) {
            auto child_nodes = gather_nodes(child);
            result.insert(result.end(), child_nodes.begin(), child_nodes.end());
        }
        return result;
    }

    void split_nodes(const std::vector<GeoNodeImpl>& flat_nodes, 
                     std::vector<GeoNodeImpl>& mesh_nodes, 
                     std::vector<GeoNodeImpl>& pc_nodes, 
                     std::vector<GeoNodeImpl>& line_nodes,
                     std::vector<GeoNodeImpl>& edge_nodes,
                     std::vector<GeoNodeImpl>& label_nodes) {
        for (const auto& node : flat_nodes) {
            if (node->mesh.is_empty() || !node->visible) continue;
            
            uint32_t state = node->material.pipeline_state;
            if (state == 4 || state == 8) {   // thick lines: strip / round joins
                line_nodes.push_back(node);
            } else if (state == 5) { // SOA Point Cloud
                pc_nodes.push_back(node);
            } else if (state == 6) { // SOA Edge Topology
                edge_nodes.push_back(node);
            } else if (state == 3) { // SOA Mesh
                mesh_nodes.push_back(node);
            } else if (state == 7) { // Text labels (LabelController)
                label_nodes.push_back(node);
            }
        }
    }
    
    std::vector<GeoNodeImpl> get_all_nodes() {
        std::vector<GeoNodeImpl> flat_nodes;
        for (auto& node : nodes) {
            auto gathered = gather_nodes(node);
            flat_nodes.insert(flat_nodes.end(), gathered.begin(), gathered.end());
        }
        return flat_nodes;
    }

public:
    simd_float3 closest_point_on_lines(simd_float3 p1, simd_float3 d1, simd_float3 p2, simd_float3 d2) {
        simd_float3 n = simd_cross(d1, d2);
        if (simd_length_squared(n) < 1e-6f) return p2;
        simd_float3 n1 = simd_cross(d1, n);
        return p2 + d2 * (simd_dot(p1 - p2, n1) / simd_dot(d2, n1));
    }

    // The event as this viewer sees it. The renderer sends rays through its own camera over the
    // whole view; a viewer with its own camera is drawn through that camera into its viewport, so
    // the cursor (now and on the previous event) is moved into the viewport's own -1..1 range and
    // the rays are rebuilt through the viewer's camera. Without a camera it's the event unchanged.
    ViewerEvent in_viewer_space(const ViewerEvent& event) const {
        if (!camera) return event;
        // viewport: top-left origin fractions; normalized_pos: whole view, y up. Its centre in
        // -1..1 is (2x + w - 1, 1 - 2y - h) and its half size is (w, h).
        auto local = [&](simd_float2 p) {
            return simd_make_float2((p.x - (2 * viewport.x + viewport.z - 1)) / viewport.z,
                                    (p.y - (1 - 2 * viewport.y - viewport.w)) / viewport.w);
        };
        ViewerEvent e = event;
        auto [ray_origin, ray_dir, pos] = ray_from_screen_space((local(event.normalized_pos) + 1) * 0.5f, simd_make_float2(1, 1), *camera);
        auto [prev_ray_origin, prev_ray_dir, prev_pos] = ray_from_screen_space((local(event.prev_normalized_pos) + 1) * 0.5f, simd_make_float2(1, 1), *camera);
        e.normalized_pos = pos;
        e.prev_normalized_pos = prev_pos;
        e.ray_origin = ray_origin;
        e.ray_dir = ray_dir;
        e.prev_ray_origin = prev_ray_origin;
        e.prev_ray_dir = prev_ray_dir;
        return e;
    }

    virtual bool handle_event(const ViewerEvent& renderer_event) {
        const ViewerEvent event = in_viewer_space(renderer_event);
        auto all_nodes = get_all_nodes();
        

        switch (event.type) {
            case MouseEvent::Tap: {
                HitResult result = ray_hit_closest(event.ray_origin, event.ray_dir, all_nodes);
                std::cout << "Tap" << std::endl;
                std::vector<GeoNodeImpl> gizmo_nodes;
                gizmo_nodes.push_back(transform_gizmo.x_axis);
                gizmo_nodes.push_back(transform_gizmo.y_axis);
                gizmo_nodes.push_back(transform_gizmo.z_axis);
                gizmo_nodes.push_back(transform_gizmo.xy_plane);
                gizmo_nodes.push_back(transform_gizmo.yz_plane);
                gizmo_nodes.push_back(transform_gizmo.zx_plane);
                HitResult gizmo_hit = ray_hit_closest(event.ray_origin, event.ray_dir, gizmo_nodes);
                if (gizmo_hit.hit) {
                    break;
                }
                if (!result.hit || !result.node->draggable) {
                    active_node = nullptr;
                    return false;
                }
                
                active_node = result.node;
                while (active_node->select_parent) {
                    GeoNodeImpl parent = active_node->parent.lock();
                    if (!parent) break;
                    active_node = parent;
                }
                std::cout << "Selected node: " << active_node->name << "\n";
                break;
            }
            case MouseEvent::Hover:
                current_drag_axis = DragAxis::None;
                break;
                
            case MouseEvent::Drag: {
                if (!active_node) {
                    return false;
                    break;
                }
                if (current_drag_axis == DragAxis::None) {
                    std::vector<GeoNodeImpl> gizmo_nodes;
                    gizmo_nodes.push_back(transform_gizmo.x_axis);
                    gizmo_nodes.push_back(transform_gizmo.y_axis);
                    gizmo_nodes.push_back(transform_gizmo.z_axis);
                    gizmo_nodes.push_back(transform_gizmo.xy_plane);
                    gizmo_nodes.push_back(transform_gizmo.yz_plane);
                    gizmo_nodes.push_back(transform_gizmo.zx_plane);
                    
                    HitResult gizmo_hit = ray_hit_closest(event.ray_origin, event.ray_dir, gizmo_nodes);
                    if (!gizmo_hit.hit) {
                        return false;
                    }
                    if (gizmo_hit.node == transform_gizmo.x_axis) current_drag_axis = DragAxis::X;
                    else if (gizmo_hit.node == transform_gizmo.y_axis) current_drag_axis = DragAxis::Y;
                    else if (gizmo_hit.node == transform_gizmo.z_axis) current_drag_axis = DragAxis::Z;
                    else if (gizmo_hit.node == transform_gizmo.xy_plane) current_drag_axis = DragAxis::XY;
                    else if (gizmo_hit.node == transform_gizmo.yz_plane) current_drag_axis = DragAxis::YZ;
                    else if (gizmo_hit.node == transform_gizmo.zx_plane) current_drag_axis = DragAxis::ZX;
                    std::cout << "Gizmo Drag axis set\n";
                }
                
                simd_float3 delta_pos = {0, 0, 0};
                
                if (current_drag_axis == DragAxis::Free) {
                    simd_float3 prev = project(event.prev_ray_origin, event.prev_ray_dir);
                    simd_float3 current = project(event.ray_origin, event.ray_dir);
                    delta_pos = current - prev;
                } else if (current_drag_axis == DragAxis::XY || current_drag_axis == DragAxis::YZ || current_drag_axis == DragAxis::ZX) {
                    simd_float3 normal = {0, 0, 0};
                    if (current_drag_axis == DragAxis::XY) normal = {0, 0, 1};
                    else if (current_drag_axis == DragAxis::YZ) normal = {1, 0, 0};
                    else if (current_drag_axis == DragAxis::ZX) normal = {0, 1, 0};
                    
                    simd_float3 origin = active_node->world_transform.SIMD_MAT(0).columns[3].xyz;
                    simd_float3 p_prev = ray_plane_dir(event.prev_ray_origin, event.prev_ray_dir, normal, origin);
                    simd_float3 p_curr = ray_plane_dir(event.ray_origin, event.ray_dir, normal, origin);
                    delta_pos = p_curr - p_prev;
                } else if (current_drag_axis != DragAxis::None) {
                    simd_float3 axis_dir = {0, 0, 0};
                    if (current_drag_axis == DragAxis::X) axis_dir = {1, 0, 0};
                    else if (current_drag_axis == DragAxis::Y) axis_dir = {0, 1, 0};
                    else if (current_drag_axis == DragAxis::Z) axis_dir = {0, 0, 1};
                    
                    simd_float3 origin = active_node->world_transform.SIMD_MAT(0).columns[3].xyz;
                    simd_float3 p_prev = closest_point_on_lines(event.prev_ray_origin, event.prev_ray_dir, origin, axis_dir);
                    simd_float3 p_curr = closest_point_on_lines(event.ray_origin, event.ray_dir, origin, axis_dir);
                    delta_pos = p_curr - p_prev;
                }
                
                active_node->local_transform.SIMD_MAT(0) = active_node->local_transform.SIMD_MAT(0) * Translation(delta_pos);
                active_node->local_transform.tape->version = ++global_epoch;   // set_local_transform keeps a tape
                break;
            }
            case MouseEvent::Zoom:
                return false;
            case MouseEvent::Scroll:
                return false;
        }
        return true;
    }

    HitResult ray_hit_closest(simd_float3 ray_origin, simd_float3 ray_dir, const std::vector<GeoNodeImpl>& test_nodes) {
        HitResult best_hit = {false, MAXFLOAT, nullptr, {0,0,0}};
        
        for (int i = 0; i < test_nodes.size(); i++) {
            if (!test_nodes[i]->draggable || !test_nodes[i]->visible) continue;
            if (test_nodes[i]->mesh.is_empty()) continue;
            if (test_nodes[i]->material.pipeline_state == 4 || test_nodes[i]->material.pipeline_state == 8) continue;
            matrix verts = test_nodes[i]->mesh.vert_position;
            matrix idx = test_nodes[i]->mesh.indices;
            verts = verts.pad(0, 1, -1, matrix::scalar(1.0f));
            
            verts.eval_metal();
            idx.eval_cpu();
            
            simd_float3* buff = (simd_float3*)verts.buffer;
            uint32_t* indices = (uint32_t*)idx.buffer;
            
            int num_instances = test_nodes[i]->world_transform.shape()[0];
            
            if (test_nodes[i]->material.pipeline_state == 4 || test_nodes[i]->material.pipeline_state == 8) {
                float line_width = test_nodes[i]->material.line_width;
                for (int inst = 0; inst < num_instances; inst++) {
                    simd_float4x4 transform = test_nodes[i]->world_transform.SIMD_MAT(inst);
                    size_t num_points = verts.shape()[0];
                    for (int j = 0; j < (int)num_points - 1; j++) {
                        simd_float3 p1 = simd_mul(transform, simd_make_float4(buff[j], 1.0f)).xyz;
                        simd_float3 p2 = simd_mul(transform, simd_make_float4(buff[j+1], 1.0f)).xyz;
                        
                        simd_float3 u = ray_dir;
                        simd_float3 v = p2 - p1;
                        simd_float3 w = ray_origin - p1;
                        float a = simd_dot(u, u);
                        float b = simd_dot(u, v);
                        float c = simd_dot(v, v);
                        float d = simd_dot(u, w);
                        float e = simd_dot(v, w);
                        float D = a*c - b*b;
                        float sc, tc;
                        
                        if (D < 1e-8f) {
                            sc = 0.0f;
                            tc = (b > c ? d / b : e / c);
                        } else {
                            sc = (b*e - c*d) / D;
                            tc = (a*e - b*d) / D;
                        }
                        
                        if (tc < 0.0f) tc = 0.0f;
                        else if (tc > 1.0f) tc = 1.0f;
                        
                        sc = (b*tc - d) / a;
                        if (sc < 0.0f) continue;
                        
                        simd_float3 dP = w + (sc * u) - (tc * v);
                        float dist = simd_length(dP);
                        
                        if (dist <= line_width) {
                            if (sc < best_hit.distance) {
                                best_hit.hit = true;
                                best_hit.distance = sc;
                                best_hit.node = test_nodes[i];
                                best_hit.hit_point = ray_origin + sc * ray_dir;
                            }
                        }
                    }
                }
                continue;
            }
            
            for (int inst = 0; inst < num_instances; inst++) {
                simd_float4x4 transform = test_nodes[i]->world_transform.SIMD_MAT(inst);
                
                size_t total_verts = idx.total_size;
                for (int j = 0; j < total_verts; j+=3) {
                    simd_float3 p1 = simd_mul(transform, simd_make_float4(buff[indices[j+0]], 1.0f)).xyz;
                    simd_float3 p2 = simd_mul(transform, simd_make_float4(buff[indices[j+1]], 1.0f)).xyz;
                    simd_float3 p3 = simd_mul(transform, simd_make_float4(buff[indices[j+2]], 1.0f)).xyz;
                    simd_float3 n = plane_norm(p1, p2, p3);
                    if (simd_length_squared(n) < 1e-14f) continue;
                    
                    simd_float3 p = ray_plane_dir(ray_origin, ray_dir, n, p1);
                    bool hit = point_inside_triangle(p, p1, p2, p3);
                    if (hit) {
                        float dist = simd_distance(ray_origin, p);
                        if (dist < best_hit.distance) {
                            best_hit.hit = true;
                            best_hit.distance = dist;
                            best_hit.node = test_nodes[i];
                            best_hit.hit_point = p;
                        }
                    }
                }
            }
        }
        return best_hit;
    }

    // The renderer's touch handlers, rerouted: the same Camera3D calls with the same arguments,
    // on this viewer's camera, or `fallback` (the renderer's) when it has none. Non-interactive
    // viewers ignore them.
    void handle_camera(float deltaX, float deltaY, bool isRightMouseButton, bool isShiftPressed,
                       TransformationMode mode, Camera3D& fallback) {
        if (!interactive) return;
        (camera ? *camera : fallback).handleMouseEvents(deltaX, deltaY, isRightMouseButton, isShiftPressed, mode);
    }

    void camera_update_old(Camera3D& fallback) {
        if (!interactive) return;
        (camera ? *camera : fallback).updateOLD();
    }

    virtual bool ray_hit(simd_float3 ray_origin, simd_float3 ray_dir) {
        HitResult res = ray_hit_closest(ray_origin, ray_dir, nodes);
        return res.hit;
    }
    
    simd_float3 project(simd_float3 ray_origin, simd_float3 ray_dir) {
        HitResult res = ray_hit_closest(ray_origin, ray_dir, nodes);
        if (res.hit) return res.hit_point;
        return {0, 0, 0};
    }
};

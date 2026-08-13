#pragma once
#include <vector>
#include <simd/simd.h>
#include "GeoNode.cpp"
#include "./../primitives.cpp"
#include "./../Mods/Utils.h"
#include "./../matrix.h"
#include "Controllers.cpp"

// Forward declarations to make code compile if necessary (conceptual)
// struct RenderBucketJob { Mesh mesh; Material material; matrix instance_buffer; uint32_t instance_count; };
// class GlobalRenderer { static void submit(const RenderBucketJob&); };

class Camera3D; // Forward declaration


bool HitTriangles(matrix triangles, simd_float3 ray_origin, simd_float3 ray_direction) {
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

bool point_inside_triangle(simd_float3 p, simd_float3 p1, simd_float3 p2, simd_float3 p3) {
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

simd_float3 plane_norm(simd_float3 p1, simd_float3 p2, simd_float3 p3) {
    return simd_cross(p2-p1, p3-p1);
}

simd_float3 ray_plane_dir(simd_float3 ray_origin, simd_float3 ray_dir, simd_float3 plane_norm, simd_float3 plane_point) {
    // r.n = d
    // r = bt + c
    
    // (bt+c).n = d
    // t = (d - c.n) / (b.n)
    float d = simd_dot(plane_point, plane_norm);
    float t = ( d - simd_dot(ray_origin, plane_norm) )/ simd_dot(ray_dir, plane_norm);
    return t * ray_dir + ray_origin;
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

    virtual ~Viewer() = default;

    virtual void draw(id<MTLRenderCommandEncoder> cmdEncoder, 
                      id<MTLDevice> metalDevice, 
                      id<MTLRenderPipelineState> __strong (&predefinedStates)[7], 
                      Camera3D* cam, 
                      int& active_state) {
        current_pass_id +=1;
        
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

        std::vector<GeoNodeImpl> mesh_nodes, pc_nodes, line_nodes, edge_nodes;
        split_nodes(scene_nodes, mesh_nodes, pc_nodes, line_nodes, edge_nodes);

        std::vector<GeoNodeImpl> gizmo_mesh_nodes, gizmo_pc_nodes, gizmo_line_nodes, gizmo_edge_nodes;
        split_nodes(gizmo_nodes, gizmo_mesh_nodes, gizmo_pc_nodes, gizmo_line_nodes, gizmo_edge_nodes);
        
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
            }
        };
        invalidate_nodes(scene_nodes);
        invalidate_nodes(line_nodes);
        invalidate_nodes(edge_nodes);
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
                ensure_metal(node->topology.edges);

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
                size_m vertexCount = (node->material.colors.dims >= 2) ? node->material.colors.shape()[node->material.colors.dims - 2] : 1;
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

                if (active_state == 4) {
                    [cmdEncoder setVertexBytes:&cam->position length:sizeof(simd_float3) atIndex:6];
                    [cmdEncoder setVertexBytes:&node->material.line_width length:sizeof(float) atIndex:7];
                    uint32_t total_points = node->mesh.vert_position.shape()[0];
                    [cmdEncoder setVertexBytes:&total_points length:sizeof(uint32_t) atIndex:8];
                }
                
                if (active_state == 6) {
                    if (node->topology.edges.metalBuffer) {
                        [cmdEncoder setVertexBuffer:(id<MTLBuffer>)node->topology.edges.metalBuffer offset:0 atIndex:8];
                    }
                    [cmdEncoder setVertexBytes:&cam->position length:sizeof(simd_float3) atIndex:6];
                    [cmdEncoder setVertexBytes:&node->material.line_width length:sizeof(float) atIndex:7];
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
                } else if (active_state == 4) { // SOA Thick Line
                    apply_bias(node->material.depth_bias);
                    uint32_t pointCount = node->mesh.vert_position.shape()[0];
                    [cmdEncoder drawPrimitives: MTLPrimitiveTypeTriangleStrip
                                   vertexStart: 0 
                                   vertexCount: pointCount * 2
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
                     std::vector<GeoNodeImpl>& edge_nodes) {
        for (const auto& node : flat_nodes) {
            if (node->mesh.is_empty()) continue;
            
            uint32_t state = node->material.pipeline_state;
            if (state == 4) {
                line_nodes.push_back(node);
            } else if (state == 5) { // SOA Point Cloud
                pc_nodes.push_back(node);
            } else if (state == 6) { // SOA Edge Topology
                edge_nodes.push_back(node);
            } else if (state == 3) { // SOA Mesh
                mesh_nodes.push_back(node);
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

    virtual bool handle_event(const ViewerEvent& event) {
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
                active_node->local_transform.tape->version = ++global_epoch;
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
            if (test_nodes[i]->mesh.is_empty()) continue;
            if (test_nodes[i]->material.pipeline_state == 4) continue;
            matrix verts = test_nodes[i]->mesh.vert_position;
            matrix idx = test_nodes[i]->mesh.indices;
            verts = verts.pad(0, 1, -1, matrix::scalar(1.0f));
            
            verts.eval_metal();
            idx.eval_cpu();
            
            simd_float3* buff = (simd_float3*)verts.buffer;
            uint32_t* indices = (uint32_t*)idx.buffer;
            
            int num_instances = test_nodes[i]->world_transform.shape()[0];
            
            if (test_nodes[i]->material.pipeline_state == 4) {
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

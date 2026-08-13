#pragma once
#include "Viewer.cpp"
#include "Controllers.cpp"
#include "matrix.h"
@import Utils;

struct Bounds2D {
    float xmin, xmax, ymin, ymax;
};

class GraphViewer : public Viewer {
public:
    GeoNodeImpl graph_node;
    GeoNodeImpl graph_frame;
    GeoNodeImpl X_axis;
    GeoNodeImpl Y_axis;
    GridController grid;
    
    float origin_x = 0.0f;
    float origin_y = 0.0f;

    GraphViewer(Bounds2D graph_space = {-10.0f, 10.0f, -10.0f, 10.0f}, Bounds2D world_space = {-1.0f, 1.0f, -1.0f, 1.0f}) {
        float w_width = world_space.xmax - world_space.xmin;
        float w_height = world_space.ymax - world_space.ymin;
        float w_cx = (world_space.xmax + world_space.xmin) / 2.0f;
        float w_cy = (world_space.ymax + world_space.ymin) / 2.0f;

        simd::float3 centre = {w_cx, w_cy, 0};
        simd::float3 size = {w_width, w_height, 200.0f};

        simd::float3 clip_min = centre - size/2;
        simd::float3 clip_max = centre + size/2;
        
        float scale_x = w_width / (graph_space.xmax - graph_space.xmin);
        float scale_y = w_height / (graph_space.ymax - graph_space.ymin);
        float translate_x = world_space.xmin - graph_space.xmin * scale_x;
        float translate_y = world_space.ymin - graph_space.ymin * scale_y;
        
        origin_x = translate_x;
        origin_y = translate_y;
        
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
        
        graph_frame = GeoNode::create("graph_frame");
        graph_frame->mesh = MeshPrimitives::quad(size.x, size.y);
        graph_frame->local_transform.SIMD_MAT(0) = Translation(centre);
        graph_frame->topology.edges = MeshPrimitives::quad_edges();
        graph_frame->material.colors = {1.0f, 1.0f, 1.0f, 1.0f};
        graph_frame->material.pipeline_state = 6;
        graph_frame->material.line_width = 0.005f;
        graph_frame->draggable = false;
        nodes.push_back(graph_frame);
        
        auto graph_back = GeoNode::create("graph_frame");
        graph_back->mesh = MeshPrimitives::quad(size.x, size.y);
        graph_back->material.colors = {0.0f, 0.0f, 0.0f, 1.0f};
        graph_back->local_transform.SIMD_MAT(0) = Translation(centre + simd_make_float3(0, 0, 0.01));
        graph_back->draggable = false;
        nodes.push_back(graph_back);

        graph_node = GeoNode::create("graph");
        graph_node->local_transform.SIMD_MAT(0) = Translation(simd_make_float3(translate_x, translate_y, 0.0f)) * Scale(simd_make_float3(scale_x, scale_y, 1.0f));
        graph_node->material.clip_min = clip_min;
        graph_node->material.clip_max = clip_max;
        graph_node->draggable = false;
        nodes.push_back(graph_node);
        // Make grid an independent node and apply clipping bounds manually
        grid.major_lines->material.clip_min = clip_min;
        grid.major_lines->material.clip_max = clip_max;
        grid.minor_lines->material.clip_min = clip_min;
        grid.minor_lines->material.clip_max = clip_max;
        nodes.push_back(grid.root_node);
    }

    void add_child(GeoNodeImpl node, simd_float3 offset) {
        // Apply the offset in the graph's local space
        node->local_transform.SIMD_MAT(0) = Translation(offset) * node->local_transform.SIMD_MAT(0);
        
        // Inherit clipping bounds from the graph node
        node->material.clip_min = graph_node->material.clip_min;
        node->material.clip_max = graph_node->material.clip_max;
        node->draggable = false;
        // Add it to the structural hierarchy
        graph_node->add_child(node);
    }

    // Override the draw method so we can steal the camera's zoom factor right before rendering!
    virtual void draw(id<MTLRenderCommandEncoder> cmdEncoder, 
                      id<MTLDevice> metalDevice, 
                      id<MTLRenderPipelineState> __strong (&predefinedStates)[7], 
                      Camera3D* cam, 
                      int& active_state) override {
        
        // 1. Calculate camera distance (zoom factor)
        float zoom_dist = simd_length(cam->target - cam->position);
        
        // 2. Base thickness for our axes (you can tweak this!)
        float base_thickness = 0.002f; 
        float dynamic_thickness = base_thickness * zoom_dist;

//        // 3. Dynamically scale the axes so they always appear the same size on screen!
//        X_axis->local_transform.SIMD_MAT(0) = Translation(simd_make_float3(0.0f, origin_y, 0.0f)) * p_scale(100.0f, dynamic_thickness, 1.0f);
//        if (X_axis->local_transform.tape) X_axis->local_transform.tape->version = ++global_epoch;
//        Y_axis->local_transform.SIMD_MAT(0) = Translation(simd_make_float3(origin_x, 0.0f, 0.0f)) * p_scale(dynamic_thickness, 100.0f, 1.0f);
//        if (Y_axis->local_transform.tape) Y_axis->local_transform.tape->version = ++global_epoch;
        
        // 4. Update the dynamic grid geometry (Aditya's logic)
        simd_float4x4 graph_mat = graph_node->local_transform.SIMD_MAT(0);
        simd_float3 graph_scale = simd_make_float3(simd_length(graph_mat.columns[0].xyz),
                                                   simd_length(graph_mat.columns[1].xyz),
                                                   simd_length(graph_mat.columns[2].xyz));
        simd_float3 graph_offset = graph_mat.columns[3].xyz;
        
        grid.update_grid(cam->position, zoom_dist, graph_scale, graph_offset);
        
        // Call the standard viewer draw loop
        Viewer::draw(cmdEncoder, metalDevice, predefinedStates, cam, active_state);
    }

    virtual bool handle_event(const ViewerEvent& event) override {
        // Allow gizmo manipulation to take priority if active
        if (Viewer::handle_event(event)) return true;

        // Ensure we only interact when the mouse is over the graph frame
        HitResult hit = ray_hit_closest(event.ray_origin, event.ray_dir, {graph_frame});
        if (!hit.hit) return false;

        if (event.type == MouseEvent::Drag || event.type == MouseEvent::Scroll) {
            // Find plane geometry in world space
            simd_float4x4 world_mat = graph_frame->world_transform.SIMD_MAT(0);
            simd_float3 normal = world_mat.columns[2].xyz; // Z axis of the frame
            simd_float3 origin = world_mat.columns[3].xyz;

            // Project rays onto the plane
            simd_float3 p_prev = ray_plane_dir(event.prev_ray_origin, event.prev_ray_dir, normal, origin);
            simd_float3 p_curr = ray_plane_dir(event.ray_origin, event.ray_dir, normal, origin);
            
            simd_float3 delta_pos = p_curr - p_prev;
            
            // Apply translation directly exactly like Viewer.cpp does for Gizmos
            graph_node->local_transform.SIMD_MAT(0) = Translation(delta_pos) * graph_node->local_transform.SIMD_MAT(0);
            graph_node->local_transform.tape->version = ++global_epoch;
            return true;
        }
        else if (event.type == MouseEvent::Zoom) {
            simd_float4x4 world_mat = graph_frame->world_transform.SIMD_MAT(0);
            simd_float3 normal = world_mat.columns[2].xyz;
            simd_float3 origin = world_mat.columns[3].xyz;
            simd_float3 p_curr = ray_plane_dir(event.ray_origin, event.ray_dir, normal, origin);
            
            // Scale around the mouse cursor in world space natively
            float scale_factor = 1.0f + event.zoom_magnitude;
            graph_node->local_transform.SIMD_MAT(0) = Translation(p_curr) * Scale(simd_make_float3(scale_factor, scale_factor, 1.0f)) * Translation(-p_curr) * graph_node->local_transform.SIMD_MAT(0);
            graph_node->local_transform.tape->version = ++global_epoch;
            return true;
        }
        
        return false;
    }

    void render_graph(const matrix& x, const matrix& y, matrix color = {1.0f, 1.0f, 1.0f, 1.0f}, float line_width = 0.0025f) {
        matrix points = matrix::stack({x, y}, -1);
        color.begin_refcount();
        // Pad with z = 0
        points = matrix::concat({points, matrix::zeros({points.shape()[0], 1})}, -1);
        
        GeoNodeImpl node_to_update = graph_node;
        
        dispatch_async(dispatch_get_main_queue(), ^{
            node_to_update->mesh.vert_position = points;
            node_to_update->mesh.indices = matrix(0, 0, dtype::UInt32); // Handled by vertex shader!
            node_to_update->material.pipeline_state = 4;
            node_to_update->material.line_width = line_width;
            node_to_update->material.colors = color;
        });
    }
};

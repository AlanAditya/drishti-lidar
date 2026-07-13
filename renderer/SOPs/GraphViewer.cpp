#pragma once
#include "Viewer.cpp"
#include "Controllers.h"
#include "matrix.h"
@import Utils;
class GraphViewer : public Viewer {
public:
    GeoNodeImpl graph_node;
    GeoNodeImpl X_axis;
    GeoNodeImpl Y_axis;

    GraphViewer() {
        X_axis = GeoNode::create("X axis");
        X_axis->mesh = MeshPrimitives::quad(1.0f, 1.0f); // Base 1x1 quad, we will scale it dynamically!
        X_axis->material.colors = {1.0f, 0.0f, 0.0f, 1.0f}; // red
        nodes.push_back(X_axis);
        
        Y_axis = GeoNode::create("Y axis");
        Y_axis->mesh = MeshPrimitives::quad(1.0f, 1.0f);
        Y_axis->material.colors = {0.0f, 1.0f, 0.0f, 1.0f};
        nodes.push_back(Y_axis);

        graph_node = GeoNode::create("graph");
        // Don't forget to push your graph_node if you use it later!
        nodes.push_back(graph_node);
    }

    // Override the draw method so we can steal the camera's zoom factor right before rendering!
    virtual void draw(id<MTLRenderCommandEncoder> cmdEncoder, 
                      id<MTLDevice> metalDevice, 
                      id<MTLRenderPipelineState> __strong (&predefinedStates)[4], 
                      Camera3D* cam, 
                      int& active_state) override {
        
        // 1. Calculate camera distance (zoom factor)
        float zoom_dist = simd_length(cam->target - cam->position);
        
        // 2. Base thickness for our axes (you can tweak this!)
        float base_thickness = 0.002f; 
        float dynamic_thickness = base_thickness * zoom_dist;

        // 3. Dynamically scale the axes so they always appear the same size on screen!
        X_axis->local_transform.SIMD_MAT(0) = p_scale(100.0f, dynamic_thickness, 1.0f);
        Y_axis->local_transform.SIMD_MAT(0) = p_scale(dynamic_thickness, 100.0f, 1.0f);
        
        // Call the standard viewer draw loop
        Viewer::draw(cmdEncoder, metalDevice, predefinedStates, cam, active_state);
    }

    // Boilerplate for your tensor logic!
    void render_graph(const matrix& x, const matrix& y, float line_width = 0.05f) {
//        auto func = [](std::vector<matrix> xy) {
//            matrix x = xy[0];
//            matrix y = xy[1];
//            size_t no_of_points = x.shape()[0];
//            float line_width = 0.005f;
//            matrix points = matrix::stack({x, y}, -1);
//            
//            matrix direction_vectors = points.slice(R(1, points.shape()[0]), 0) - points.slice(R(0, points.shape()[0] - 1), 0);
//            
//            matrix norm_direction_vectors = direction_vectors /  matrix::sqrt((direction_vectors * direction_vectors).sum(-1, true));
//            
//            matrix pool_norm_direction_vectors = norm_direction_vectors.slice(R(1, points.shape()[0] - 1), 0) + norm_direction_vectors.slice(R(0, points.shape()[0] - 2), 0);
//            
//            norm_direction_vectors = matrix::concat({norm_direction_vectors[R(0, 1)], pool_norm_direction_vectors, norm_direction_vectors[R(-1, LLONG_MAX)]}, 0);
//            norm_direction_vectors = norm_direction_vectors / matrix::sqrt((norm_direction_vectors * norm_direction_vectors).sum(-1, true));
//            
//            matrix dx = norm_direction_vectors.slice(R(0, 1), 1);
//            matrix dy = norm_direction_vectors.slice(R(1, 2), 1);
//            matrix perp_vectors = matrix::concat({dy * -1.0f, dx}, 1);
//            
//            matrix layer1 = points + perp_vectors * line_width;
//            matrix layer2 = points - perp_vectors * line_width;
//            matrix inter_leave = matrix::concat({layer1, layer2}, -1).reshape(2 * points.shape()[0], points.shape()[1]);
//            
//            
//            inter_leave = matrix::concat({inter_leave, matrix::zeros({2 * points.shape()[0], 1})}, -1);
//            return std::vector<matrix>{ inter_leave };
//        };
//        
//        auto f = matrix::multi_jit_graph_gpu(func, {x.zeros(), y.zeros()});
//        for (int i = 0; i < 1000000; i++) {
//            
//            @autoreleasepool {
//                Timer time;
//                std::vector<matrix> outp = f({const_cast<matrix&>(x), const_cast<matrix&>(y)});
//                outp[0].eval_metal();
//            }
//            
//        }
        size_t no_of_points = x.shape()[0];
        matrix points = matrix::stack({x, y}, -1);
        
        matrix direction_vectors = points.slice(R(1, points.shape()[0]), 0) - points.slice(R(0, points.shape()[0] - 1), 0);
        
        matrix norm_direction_vectors = direction_vectors /  matrix::sqrt((direction_vectors * direction_vectors).sum(-1, true));
        
        matrix pool_norm_direction_vectors = norm_direction_vectors.slice(R(1, points.shape()[0] - 1), 0) + norm_direction_vectors.slice(R(0, points.shape()[0] - 2), 0);
        // mag 2 (0*), mag sqrt(2) (90*)
        
        matrix pool_mag_sq = ((pool_norm_direction_vectors * pool_norm_direction_vectors).sum(-1, true));
        // mag 1 (0*), mag 1 (90*)
        matrix norm_pool = 2 * pool_norm_direction_vectors / pool_mag_sq;
        // mag 1 (0*), mag 1 (90*)
        
        // mag 1 (0*), mag sqrt(2) (90*)
        // trick to get lower the angle higher the mgnitude so cornors look proportionate when we make these direction vectors perpendiculare and add them as offset
        
        norm_direction_vectors = matrix::concat({norm_direction_vectors[R(0, 1)], norm_pool, norm_direction_vectors[R(-1, LLONG_MAX)]}, 0);
        
        matrix dx = norm_direction_vectors.slice(R(0, 1), 1);
        matrix dy = norm_direction_vectors.slice(R(1, 2), 1);
        matrix perp_vectors = matrix::concat({dy * -1.0f, dx}, 1);
        
        matrix layer1 = points + perp_vectors * line_width;
        matrix layer2 = points - perp_vectors * line_width;
        matrix inter_leave = matrix::concat({layer1, layer2}, -1).reshape(2 * points.shape()[0], points.shape()[1]);
        
        
        inter_leave = matrix::concat({inter_leave, matrix::zeros({2 * points.shape()[0], 1})}, -1);
        
        // map (min, max)=>(-1, 1) for uv
        Mesh LineGraph;
        LineGraph.vert_position = inter_leave;
        LineGraph.uv_coords = inter_leave;
        matrix indices(3, (no_of_points - 1) * 2 * 3, dtype::UInt32);
        indices.shape()[0] = (no_of_points - 1);
        indices.shape()[1] = 2;
        indices.shape()[2] = 3;
        indices.calcStrides();
        
        for (int i = 0; i < no_of_points-1; i++) {
            indices.at<uint32_t>(i, 0, 0) = 2 * (i); // layer 1
            indices.at<uint32_t>(i, 0, 1) = 2 * (i) + 1; // layer 2
            indices.at<uint32_t>(i, 0, 2) = 2 * (i + 1); // layer 1 next
            
            indices.at<uint32_t>(i, 1, 0) = 2 * (i+1); // layer 1 next
            indices.at<uint32_t>(i, 1, 1) = 2 * (i)+1; // layer 2
            indices.at<uint32_t>(i, 1, 2) = 2 * (i+1)+1; // layer 2 next
        }
        
        LineGraph.indices = indices;
        
        // Capture the shared_ptr explicitly so it stays alive even if the GraphViewer is destroyed.
        GeoNodeImpl node_to_update = graph_node;
        
        // Dispatch the mesh assignment to the main queue to avoid a data race
        // with the main rendering thread accessing graph_node->mesh during Viewer::draw()
        dispatch_async(dispatch_get_main_queue(), ^{
            node_to_update->mesh = LineGraph;
        });
    }
};

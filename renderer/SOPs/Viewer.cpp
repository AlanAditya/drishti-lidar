#include <vector>
#include "GeoNode.cpp"
#include "./../primitives.cpp"

// Forward declarations to make code compile if necessary (conceptual)
// struct RenderBucketJob { Mesh mesh; Material material; matrix instance_buffer; uint32_t instance_count; };
// class GlobalRenderer { static void submit(const RenderBucketJob&); };

class Camera3D; // Forward declaration

class Viewer {
public:
    uint64_t current_pass_id = 0;
    std::vector<GeoNodeImpl> nodes; // The nodes managed by this viewer

    virtual ~Viewer() = default;

    virtual void draw(id<MTLRenderCommandEncoder> cmdEncoder, 
                      id<MTLDevice> metalDevice, 
                      id<MTLRenderPipelineState> __strong (&predefinedStates)[4], 
                      Camera3D* cam, 
                      int& active_state) {
        current_pass_id +=1;
        std::vector<GeoNodeImpl> flat_nodes;
        for (auto& node : nodes) {
            auto gathered = gather_nodes(node);
            flat_nodes.insert(flat_nodes.end(), gathered.begin(), gathered.end());
        }

        for (const auto& node : flat_nodes) {
            if (node->mesh.is_empty()) continue;
            
            if (node->mesh.vert_position.tape) node->mesh.vert_position.tape->invalidate_pass(current_pass_id);
            if (node->world_transform.tape) node->world_transform.tape->invalidate_pass(current_pass_id);
            if (node->material.colors.tape) node->material.colors.tape->invalidate_pass(current_pass_id);
            if (node->mesh.uv_coords.tape) node->mesh.uv_coords.tape->invalidate_pass(current_pass_id);
            if (node->mesh.indices.tape) node->mesh.indices.tape->invalidate_pass(current_pass_id);

            node->mesh.vert_position.eval();
            node->world_transform.eval();
            node->material.colors.eval();
            node->mesh.uv_coords.eval();
            node->mesh.indices.eval();

            auto ensure_metal = [](matrix& mat) {
                if (mat.total_size > 0 && mat.metalBuffer == nullptr) {
                    mat.buildMetalBuffer();
                }
            };
            
            ensure_metal(node->mesh.vert_position);
            ensure_metal(node->world_transform);
            ensure_metal(node->material.colors);
            ensure_metal(node->mesh.uv_coords);
            ensure_metal(node->mesh.indices);

            if (active_state != 3) {
                [cmdEncoder setRenderPipelineState:predefinedStates[3]];
                active_state = 3;
            }

            if (node->mesh.vert_position.metalBuffer) {
                [cmdEncoder setVertexBuffer:(id<MTLBuffer>)node->mesh.vert_position.metalBuffer offset:0 atIndex:0];
            }
            if (node->world_transform.metalBuffer) {
                [cmdEncoder setVertexBuffer:(id<MTLBuffer>)node->world_transform.metalBuffer offset:0 atIndex:1];
            }
            
            [cmdEncoder setVertexBytes:&cam->viewMatrix length:sizeof(simd_float4x4) atIndex:2];
            
            uint32_t color_stride = 0;
            if (node->material.colors.metalBuffer) {
                // If there's only 1 color, force stride to 0 so all vertices sample index 0
                if (node->material.colors.shape()[0] == 1) {
                    color_stride = 0;
                } else {
                    color_stride = node->material.colors.strides()[0] / 4;
                }
                [cmdEncoder setVertexBuffer:(id<MTLBuffer>)node->material.colors.metalBuffer offset:0 atIndex:3];
            }
            [cmdEncoder setVertexBytes:&color_stride length:sizeof(uint32_t) atIndex:4];
            
            if (node->mesh.uv_coords.metalBuffer) {
                [cmdEncoder setVertexBuffer:(id<MTLBuffer>)node->mesh.uv_coords.metalBuffer offset:0 atIndex:5];
            }

            bool isTextured = node->material.has_texture;
            [cmdEncoder setFragmentBytes:&isTextured length:sizeof(bool) atIndex:0];
            if (isTextured && node->material.texture) {
                [cmdEncoder setFragmentTexture:node->material.texture atIndex:0];
            }
            
            uint32_t indexCount = node->mesh.indices.total_size;
            uint32_t instanceCount = node->world_transform.shape()[0];
            
            if (node->mesh.indices.metalBuffer) {
                [cmdEncoder drawIndexedPrimitives: MTLPrimitiveTypeTriangle
                                       indexCount: indexCount 
                                        indexType: MTLIndexTypeUInt32 
                                      indexBuffer: node->mesh.indices.metalBuffer
                                indexBufferOffset: 0
                                    instanceCount: instanceCount];
            }
        }
    }

    void render_scene(const GeoNodeImpl& root) {
        current_pass_id++;

        // 1. Gather all nodes (Flat list via simple DFS/BFS)
        std::vector<GeoNodeImpl> flat_nodes = gather_nodes(root);

        // 2. Evaluate and Submit
        for (const auto& node : flat_nodes) {
            
            // Skip Null/Transform Groups
            if (node->mesh.is_empty()) continue;

//            RenderBucketJob job;
//            job.mesh = node->mesh;
//            job.material = node->material;
//            
//            // The DAG handles the 2-pass lazy evaluation under the hood.
//            // If time, parent offsets, or local instances changed, it computes it here.
//            job.instance_buffer = node->world_transform.evaluate_metal(current_pass_id); 
//            job.instance_count = node->world_transform.size(0);
//            
//            GlobalRenderer.submit(job);
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
};

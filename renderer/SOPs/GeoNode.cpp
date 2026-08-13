#pragma once
#include <memory>
#include <vector>
#include <string>
#include "./../matrix.h" // Assuming matrix handles are here

struct DepthBias {
    bool custom = false;
    float bias = 0.0f;
    float slopeScale = 0.0f;
    float clamp = 0.0f;
};

// 1. MATERIAL
struct Material {
    // Shape: [N, 4] for per-vertex color mapping or [1, 4] for uniform color
    matrix colors; 
    
    id<MTLTexture> texture = nil;
    bool has_texture = false;
    
    DepthBias depth_bias;
    uint32_t pipeline_state = 3;
    float line_width = 0.05f;
    simd_float3 clip_min = {-1e5f, -1e5f, -1e5f}; // Default: infinite bounds
    simd_float3 clip_max = { 1e5f,  1e5f,  1e5f};

    Material() : colors(matrix::of<float>({1.0f, 1.0f, 1.0f, 1.0f}).reshape(1, 4)) {
    }
};

// 2. MESH
struct Mesh {
    matrix vert_position; // Shape: [V, 3] or [V, 4] Tensor Handle
    matrix uv_coords;     // Shape: [V, 2] Tensor Handle
    matrix indices;       // Shape: [I] Tensor Index Handle
    
    Mesh() : vert_position(0, 0, dtype::Float), uv_coords(0, 0, dtype::Float), indices(0, 0, dtype::UInt32) {}

    // An empty tensor signifies a purely virtual "Transform/Null Node"
    bool is_empty() const {
        return vert_position.total_size == 0;
    }
};

class GeoNode;
using GeoNodeImpl = std::shared_ptr<GeoNode>;

struct Topology {
    matrix edges; // Shape: [E, 2], dtype: UInt32
    
    Topology() : edges(0, 0, dtype::UInt32) {}
    
    bool is_empty() const {
        return edges.total_size == 0;
    }
};

class GeoNode : public std::enable_shared_from_this<GeoNode> {
public:
    std::string name;
    
    // THE SPACE (Matrix DAG Handles)
    // Always a 3D Tensor [Instances, 4, 4]. Defaults to [1, 4, 4].
    matrix local_transform; 
    matrix world_transform; 
    
    // THE VISUALS
    Mesh mesh;
    Topology topology;
    Material material;
    
    // INTERACTION
    bool draggable = true;
    
    // THE HIERARCHY
    std::weak_ptr<GeoNode> parent;
    std::vector<GeoNodeImpl> children;

    // Private constructor: Forces heap allocation via static factory
    private:
        explicit GeoNode(std::string node_name) 
            : name(std::move(node_name)),
              local_transform(matrix::eye(4).reshape(1, 4, 4)),
              world_transform(local_transform) 
        {
        }

    public:
        // Factory Method: Ensures we always get a GeoNodeImpl (shared_ptr)
        // (This is where your Custom Arena Allocator will hook in later)
        static GeoNodeImpl create(std::string name) {
            return std::shared_ptr<GeoNode>(new GeoNode(std::move(name)));
        }

        // Structural Mutation API
        void add_child(const GeoNodeImpl& child);
        
        void set_local_transform(const matrix& new_transform) {
            this->local_transform = new_transform;
            update_world_transform_recursive();
        }
        
        // Use this when you mutate local_transform's buffer directly (e.g., via SIMD_MAT)
        // to force the downstream DAG nodes (like world_transform) to re-evaluate.
        void invalidate() {
            world_transform.clear_trace_checks();
            for (auto& child : children) {
                child->invalidate();
            }
        }
        
    
        void update_world_transform_link();
        void update_world_transform_recursive();
};

void GeoNode::add_child(const GeoNodeImpl& child) {
    // 1. Break existing parent bond if it exists
    if (auto current_parent = child->parent.lock()) {
        auto& siblings = current_parent->children;
        siblings.erase(std::remove(siblings.begin(), siblings.end(), child), siblings.end());
    }
    
    // 2. Establish structural heap pointers
    child->parent = shared_from_this();
    children.push_back(child);

    // 3. Forge the Matrix DAG Data Link and propagate
    child->update_world_transform_recursive();
}

void GeoNode::update_world_transform_link() {
    if (auto p = parent.lock()) {
        // Step 1 & 2: Unsqueeze for combinatorial broadcasting [1, M]
        matrix parent_broad = p->world_transform.unsqueeze(1); // [P, 1, 4, 4]
        matrix child_broad  = this->local_transform.unsqueeze(0); // [1, C, 4, 4]
        
        // Step 3: Broadcasted Matrix Multiplication -> [P, C, 4, 4]
        // Note: Engine uses row vectors (v * M), so child transform is applied first: M_combined = M_local * M_parent
        matrix combined = child_broad.dot(parent_broad);
        
        // Step 4: Flatten the instance dimensions -> [P * C, 4, 4]
        this->world_transform = combined.flatten(0, 1);
    } else {
        this->world_transform = this->local_transform;
    }
}

void GeoNode::update_world_transform_recursive() {
    update_world_transform_link();
    for (auto& child : children) {
        child->update_world_transform_recursive();
    }
}

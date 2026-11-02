#pragma once
#include <memory>
#include <vector>
#include <string>
#include "./../matrix/matrix.h" // Assuming matrix handles are here

struct DepthBias {
    bool custom = false;
    float bias = 0.0f;
    float slopeScale = 0.0f;
    float clamp = 0.0f;
};

// How the mesh pipeline (3) colours a surface.
enum class Shading : uint32_t {
    Unlit = 0,   // the colour / texture as is (the default; 2D shapes, UI, plots, images)
    Lit = 1,     // diffuse + specular + ambient from the scene's lights, using the mesh normals (Blinn-Phong,
                 // three.js MeshPhongMaterial)
    Standard = 2,   // physically based, metallic/roughness (three.js MeshStandardMaterial)
};

// How a Viewer maps lit HDR levels (> 1 where lights overlap or are bright) onto the screen.
enum class ToneMapping : uint32_t {
    None = 0,   // clip at 1
    ACES = 1,      // filmic roll-off, three.js's ACESFilmicToneMapping curve (desaturates bright colours)
    Neutral = 2,   // highlight compression of Khronos PBR Neutral (three.js's NeutralToneMapping), without its dark "toe":
                   // colours below ~0.76 are untouched, only highlights compress, base colours stay true
};

// Distance fog: blends surfaces toward a colour with distance from the camera (three.js Fog / FogExp2).
struct Fog {
    enum class Mode : uint32_t { None = 0, Linear = 1, Exp2 = 2 };
    Mode mode = Mode::None;
    simd_float3 color = {0.0f, 0.0f, 0.0f};   // usually the background colour
    float near = 5.0f;       // Linear: fog starts here...
    float far = 15.0f;       // ...and is total here (smoothstep between, as three.js does)
    float density = 0.1f;    // Exp2: 1 - exp(-(density * distance)^2)
};

// A light for Shading::Lit meshes (a Viewer holds them in `lights`).
struct Light {
    enum class Type : uint32_t { Directional = 0, Point = 1 };
    Type type = Type::Directional;
    simd_float3 direction = {0.5f, -0.8f, 0.6f};   // Directional: the way the light travels (world space)
    simd_float3 position = {0.0f, 2.0f, 0.0f};     // Point: where it is (world space)
    simd_float3 color = {1.0f, 1.0f, 1.0f};
    float intensity = 1.0f;
    float range = 2.0f;                            // Point: distance at which it falls to half strength
    bool casts_shadows = false;                    // Directional: gets a shadow map (the first such light). Opt-in.

    static Light directional(simd_float3 direction, float intensity = 1.0f, simd_float3 color = simd_make_float3(1.0f, 1.0f, 1.0f)) {
        Light l;
        l.type = Type::Directional;
        l.direction = direction;
        l.intensity = intensity;
        l.color = color;
        return l;
    }
    static Light point(simd_float3 position, float intensity = 1.0f, simd_float3 color = simd_make_float3(1.0f, 1.0f, 1.0f), float range = 2.0f) {
        Light l;
        l.type = Type::Point;
        l.position = position;
        l.intensity = intensity;
        l.color = color;
        l.range = range;
        return l;
    }
};

// Light as the fragment shader reads it (DAGNodeFragmentShader's GPULight): three float4s.
struct GPULight {
    simd_float4 position_type;     // xyz position, w type (0 directional, 1 point)
    simd_float4 direction_range;   // xyz direction (directional), w range (point)
    simd_float4 color_intensity;   // rgb colour, a intensity
};

// 1. MATERIAL
struct Material {
    // Shape: [N, 4] for per-vertex color mapping or [1, 4] for uniform color
    matrix colors; 
    
    id<MTLTexture> texture = nil;
    bool has_texture = false;
    bool texture_repeat = false;    // UVs outside [0, 1] tile the texture (loaded models) instead of clamping to its edge
    
    DepthBias depth_bias;
    uint32_t pipeline_state = 3;
    Shading shading = Shading::Unlit;
    float specular = 0.35f;    // Lit: highlight strength
    float shininess = 48.0f;   // Lit: highlight tightness (Blinn-Phong exponent)
    float metallic = 0.0f;     // Standard: 0 dielectric (plastic, paint, wood), 1 metal. The colour is the metal's reflectance.
    float roughness = 0.5f;    // Standard: 0 mirror-smooth, 1 fully matte
    bool casts_shadows = true;      // drawn into shadow maps
    bool receives_shadows = true;   // Lit: darkened where the shadow-casting light is blocked
    bool tone_mapped = true;        // Lit: goes through the Viewer's tone mapping (three.js's material.toneMapped)
    bool fog = true;                // affected by the Viewer's fog (three.js's material.fog)
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
    matrix normals;       // Shape: [V, 3], unit length, object space. Empty: lit shading falls back to flat face normals
    matrix glyphs;        // Shape: [G, 16], text labels (pipeline 7) only: one glyph/plate record per quad, see vertex_label

    Mesh() : vert_position(0, 0, dtype::Float), uv_coords(0, 0, dtype::Float), indices(0, 0, dtype::UInt32),
             normals(0, 0, dtype::Float), glyphs(0, 0, dtype::Float) {}

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
    bool visible = true;   // false: the Viewer skips it (not evaluated, drawn or hit-tested); its children still draw
    bool select_parent = false;   // a tap on it selects its parent instead (the parts of a loaded model move as one)
    
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
        
        // A tapeless transform (host data, e.g. transform_of) becomes a leaf, so local_transform
        // always has a tape: after an in-place edit (SIMD_MAT), bump
        // local_transform.tape->version = ++global_epoch and the Viewer's invalidate_pass
        // re-runs only the dirty path down to the world transforms.
        void set_local_transform(const matrix& new_transform) {
            this->local_transform = new_transform;
            if (!this->local_transform.tape) this->local_transform.make_leaf();
            update_world_transform_recursive();
        }
        
    
        void update_world_transform_link();
        void update_world_transform_recursive();
};

inline void GeoNode::add_child(const GeoNodeImpl& child) {
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

inline void GeoNode::update_world_transform_link() {
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

inline void GeoNode::update_world_transform_recursive() {
    update_world_transform_link();
    for (auto& child : children) {
        child->update_world_transform_recursive();
    }
}

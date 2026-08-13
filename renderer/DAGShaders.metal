#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct VertexLineOut {
    float4 position [[position]];
    float4 color;
    float3 local_pos;
};

vertex VertexLineOut vertex_line3d(
    uint vertexID [[vertex_id]],
    constant packed_float3* points [[buffer(0)]],
    constant float4x4* instance_buffer [[buffer(1)]],
    constant float4x4& vMatrix [[buffer(2)]],
    constant float4* colors [[buffer(3)]],
    constant uint2& color_strides [[buffer(4)]],
    constant packed_float3& cam_pos [[buffer(6)]],
    constant float& line_width [[buffer(7)]],
    constant uint& total_points [[buffer(8)]],
    constant float4x4* local_transform_buffer [[buffer(9)]],
    constant uint& local_instances [[buffer(10)]],
    uint instanceId [[instance_id]]
) {
    VertexLineOut vertOut;
    
    // Each point generates 2 vertices (left and right)
    uint point_idx = vertexID / 2;
    float side = (vertexID % 2 == 0) ? -1.0 : 1.0;
    
    // Safety check
    if (point_idx >= total_points) {
        vertOut.position = float4(0,0,0,0);
        return vertOut;
    }
    
    float3 p = points[point_idx];
    
    // Pen-lift check (NaN point collapses the quad)
    if (isnan(p.x)) {
        vertOut.position = float4(0,0,0,0);
        return vertOut;
    }
    
    // Calculate position for fragment clipping bounds in world space
    vertOut.local_pos = (instance_buffer[instanceId] * float4(p, 1.0)).xyz;
    
    // Compute direction to next or previous point
    float3 dir;
    bool has_next = (point_idx < total_points - 1) && !isnan(points[point_idx + 1].x);
    bool has_prev = (point_idx > 0) && !isnan(points[point_idx - 1].x);
    
    float3 p_next = has_next ? float3(points[point_idx + 1]) : p;
    float3 p_prev = has_prev ? float3(points[point_idx - 1]) : p;
    
    // Map points to world space BEFORE computing directions to ensure non-uniform scaling works
    float3 world_p = (instance_buffer[instanceId] * float4(p, 1.0)).xyz;
    float3 world_p_next = (instance_buffer[instanceId] * float4(p_next, 1.0)).xyz;
    float3 world_p_prev = (instance_buffer[instanceId] * float4(p_prev, 1.0)).xyz;
    
    if (has_next) {
        dir = normalize(world_p_next - world_p);
    } else if (has_prev) {
        dir = normalize(world_p - world_p_prev);
    } else {
        dir = float3(1, 0, 0); 
    }
    
    // Exact view direction for this vertex
    float3 view_dir = normalize(cam_pos - world_p);
    
    // Perpendicular extrusion in world space (naturally tapers to 0 when pointing at camera)
    float3 right = cross(dir, view_dir);
    
    // Scale thickness by distance to camera to maintain constant screen-space pixel width
    float dist = length(cam_pos - world_p);
    float3 offset_p = world_p + right * line_width * dist * side;
    
    // Apply camera projection * view matrix
    vertOut.position = vMatrix * float4(offset_p, 1.0);
    
    // Apply color
    uint c_idx = instanceId * color_strides.x + point_idx * color_strides.y;
    vertOut.color = colors[c_idx];
    
    return vertOut;
}

fragment float4 fragment_line3d(VertexLineOut inColor [[stage_in]],
                                constant float3& clip_min [[buffer(1)]],
                                constant float3& clip_max [[buffer(2)]]) {
    // Clip against the bounds box mapped in local space
    if (any(inColor.local_pos < clip_min) || any(inColor.local_pos > clip_max)) {
        discard_fragment();
    }
    return inColor.color;
}

struct DAGPointCloudVertexOut {
    float4 position [[position]];
    float4 color;
    float point_size [[point_size]];
    float3 local_pos;
};

vertex DAGPointCloudVertexOut vertex_dag_pointcloud(
    uint vertexID [[vertex_id]],
    constant packed_float3* points [[buffer(0)]],
    constant float4x4* instance_buffer [[buffer(1)]],
    constant float4x4& vMatrix [[buffer(2)]],
    constant float4* colors [[buffer(3)]],
    constant uint2& color_strides [[buffer(4)]],
    constant float4x4* local_transform_buffer [[buffer(9)]],
    constant uint& local_instances [[buffer(10)]],
    uint instanceId [[instance_id]]
) {
    DAGPointCloudVertexOut vertOut;
    
    float3 p = points[vertexID];
    
    // Calculate position for fragment clipping bounds in world space
    vertOut.local_pos = (instance_buffer[instanceId] * float4(p, 1.0)).xyz;
    
    // Instance buffer handles world transform
    float4 world_pos = instance_buffer[instanceId] * float4(p, 1.0);
    
    // Apply camera projection * view matrix
    vertOut.position = vMatrix * world_pos;
    
    // Apply color
    uint c_idx = instanceId * color_strides.x + vertexID * color_strides.y;
    vertOut.color = colors[c_idx];
    
    // Set a constant point size for now, this could also be passed via a buffer
    vertOut.point_size = 5.0;
    
    return vertOut;
}

fragment float4 fragment_dag_pointcloud(DAGPointCloudVertexOut inColor [[stage_in]],
                                        float2 pointCoord [[point_coord]],
                                        constant float3& clip_min [[buffer(1)]],
                                        constant float3& clip_max [[buffer(2)]]) {
    // Clip against the bounds box mapped in local space
    if (any(inColor.local_pos < clip_min) || any(inColor.local_pos > clip_max)) {
        discard_fragment();
    }
    
    // Basic circular point
    float dist = length(pointCoord - float2(0.5));
    if (dist > 0.5) {
        discard_fragment();
    }
    return inColor.color;
}

vertex VertexLineOut vertex_edge3d(
    uint vertexID [[vertex_id]],
    constant packed_float3* vertices [[buffer(0)]],
    constant float4x4* instance_buffer [[buffer(1)]],
    constant float4x4& vMatrix [[buffer(2)]],
    constant float4* colors [[buffer(3)]],
    constant uint2& color_strides [[buffer(4)]],
    constant packed_float3& cam_forward [[buffer(6)]],
    constant float& line_width [[buffer(7)]],
    constant uint2* edges [[buffer(8)]],
    uint instanceId [[instance_id]]
) {
    VertexLineOut vertOut;
    
    // We draw 6 vertices per edge (a quad made of 2 triangles)
    uint edge_idx = vertexID / 6;
    uint v_in_quad = vertexID % 6; // 0, 1, 2, 2, 3, 0 mapping to quad corners
    
    uint2 edge = edges[edge_idx];
    float3 p0 = vertices[edge[0]];
    float3 p1 = vertices[edge[1]];
    
    float4 world_p0 = instance_buffer[instanceId] * float4(p0, 1.0);
    float4 world_p1 = instance_buffer[instanceId] * float4(p1, 1.0);
    
    // Direction of the edge in world space
    float3 dir = normalize(world_p1.xyz - world_p0.xyz);
    
    int corner = 0;
    if (v_in_quad == 0) corner = 0;
    else if (v_in_quad == 1) corner = 1;
    else if (v_in_quad == 2) corner = 2;
    else if (v_in_quad == 3) corner = 2;
    else if (v_in_quad == 4) corner = 3;
    else if (v_in_quad == 5) corner = 0;
    
    float3 world_p = (corner == 0 || corner == 3) ? world_p0.xyz : world_p1.xyz;
    float side = (corner == 0 || corner == 1) ? -1.0 : 1.0;
    
    // Exact view direction for this vertex
    float3 view_dir = normalize(cam_forward - world_p); // Note: cam_forward buffer is now bound to cam->position!
    
    // Perpendicular extrusion in world space.
    // By NOT normalizing the cross product, the length of 'right' is sin(theta).
    // This naturally tapers the line thickness to 0 when it points directly at the camera,
    // gracefully hiding the sharp quad corners inside the joint circles!
    float3 right = cross(dir, view_dir);
    
    // Scale thickness by distance to camera to maintain constant screen-space pixel width
    float dist = length(cam_forward - world_p);
    float3 offset_p = world_p + right * line_width * dist * side;
    
    vertOut.position = vMatrix * float4(offset_p, 1.0);
    
    // Color mapping
    uint c_idx = instanceId * color_strides.x + edge_idx * color_strides.y;
    vertOut.color = colors[c_idx];
    
    return vertOut;
}

fragment float4 fragment_edge3d(VertexLineOut inColor [[stage_in]]) {
    return inColor.color;
}

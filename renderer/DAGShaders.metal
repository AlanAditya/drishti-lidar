#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct VertexLineOut {
    float4 position [[position]];
    float4 color;
    float3 local_pos;
    float2 disc_uv;   // vertex_line3d's round joins: where in the disc (outside past length 1), 0 on the strip
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
    vertOut.disc_uv = float2(0);
    vertOut.position = float4(0);   // degenerate unless set below
    
    // Plain triangles, drawn with 12 * N vertices (Viewer.cpp). Each line is a pipe's outline:
    // - Vertices 0 .. 6N - 1: segment k (point k to k + 1) is a quad of 6 vertices, extruded by the
    //   half width perpendicular to the segment as seen on screen, so its width is constant
    //   whichever way it points (segment N - 1 doesn't exist and stays degenerate).
    // - From 6N on: 6 per point, a screen-aligned square that fragment_line3d trims to a disc of
    //   the half width. The discs round every joint and both ends.
    // Segment quads don't share vertices: a strip extruding each point once (along the outgoing
    // segment) twisted at corners and pinched to zero width, and a mitered one spikes at sharp
    // turns. Quads plus discs are exactly a round join.
    bool is_disc = vertexID >= 6 * total_points;
    uint corner = vertexID % 6;
    uint segment = vertexID / 6;
    // Segment quad corners: (which end, which side), as two triangles
    const uint ends[6] = {0, 0, 1, 1, 0, 1};
    const float sides[6] = {-1, 1, -1, -1, 1, 1};
    uint point_idx = is_disc ? (vertexID - 6 * total_points) / 6 : segment + ends[corner];
    
    // Safety check: past the end, or a segment without a second point
    if (point_idx >= total_points || (!is_disc && segment + 1 >= total_points)) {
        return vertOut;
    }
    
    float3 p = points[point_idx];
    
    // Pen-lift check (a NaN point drops its disc and both segments touching it)
    if (isnan(p.x) || (!is_disc && (isnan(points[segment].x) || isnan(points[segment + 1].x)))) {
        return vertOut;
    }
    
    // Calculate position for fragment clipping bounds in world space
    vertOut.local_pos = (instance_buffer[instanceId] * float4(p, 1.0)).xyz;
    float3 world_p = vertOut.local_pos;
    
    // The screen plane: perpendicular to the camera's forward axis, not to this vertex's own view
    // ray. Off-centre that ray is tilted, so extruding across it also moves the two edges in depth,
    // and on a 2D graph the far edge went behind the frame's backing quad (thick lines tapered
    // towards the sides). For a perspective vMatrix (projection * view) the w row is the forward
    // axis; an orthographic one has none, so fall back to the view ray.
    float3 view_dir = normalize(cam_pos - world_p);
    float3 forward = float3(vMatrix[0][3], vMatrix[1][3], vMatrix[2][3]);
    bool has_forward = length_squared(forward) > 1e-12;
    float3 facing = has_forward ? normalize(forward) : view_dir;
    
    // Half width in world units, scaled by depth for a constant screen-space pixel width
    float dist = has_forward ? abs(dot(cam_pos - world_p, facing)) : length(cam_pos - world_p);
    float half_width = line_width * dist;
    
    float3 offset;
    if (is_disc) {
        const float2 uvs[6] = {float2(-1, -1), float2(1, -1), float2(-1, 1), float2(-1, 1), float2(1, -1), float2(1, 1)};
        float2 uv = uvs[corner];
        // The camera's right and up axes: rows 0 and 1 of projection * view, up to scale
        float3 screen_x = normalize(float3(vMatrix[0][0], vMatrix[1][0], vMatrix[2][0]));
        float3 screen_y = normalize(float3(vMatrix[0][1], vMatrix[1][1], vMatrix[2][1]));
        offset = (screen_x * uv.x + screen_y * uv.y) * half_width;
        vertOut.disc_uv = uv;
    } else {
        // Map points to world space BEFORE computing directions to ensure non-uniform scaling works
        float3 a = (instance_buffer[instanceId] * float4(float3(points[segment]), 1.0)).xyz;
        float3 b = (instance_buffer[instanceId] * float4(float3(points[segment + 1]), 1.0)).xyz;
        // The segment as seen: projected onto the screen plane. One pointing straight at the
        // camera is a dot there; its endpoints' discs cover it.
        float3 dir = b - a;
        dir -= dot(dir, facing) * facing;
        if (length_squared(dir) < 1e-12) {
            return vertOut;
        }
        // dir and facing are perpendicular unit vectors, so this is unit length
        float3 right = cross(normalize(dir), facing);
        offset = right * half_width * sides[corner];
    }
    
    // Apply camera projection * view matrix
    vertOut.position = vMatrix * float4(world_p + offset, 1.0);
    
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
    // Round-join discs: trim the square to its circle
    if (length_squared(inColor.disc_uv) > 1.0) {
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
    constant float& point_size [[buffer(7)]],
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
    
    // Read point size from buffer(7) (mapped to line_width in CPU)
    vertOut.point_size = point_size > 0.0 ? point_size : 5.0; // Fallback to 5.0 if 0
    
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

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

struct VertexLineOut {
    float4 position [[position]];
    float4 color;
    float3 local_pos;
    float2 disc_uv;   // vertex_line3d_round: where in a joint's disc (outside past length 1), else 0
};

// Pipeline 4: thick line as one triangle strip, 2 vertices per point (Viewer draws 2N). The
// cheap line: constant screen width and mitered joints, but sharp turns spike. Pipeline 8
// (vertex_line3d_round, below) draws round joins and caps for 6x the vertices.
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
    vertOut.disc_uv = float2(0);   // fragment_line3d's disc test never discards the strip
    
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
    
    // Extrude in the screen plane: perpendicular to the camera's forward axis, not to this
    // vertex's own view ray. Off-centre that ray is tilted, so extruding across it also moves the
    // two edges in depth, and on a 2D graph the far edge went behind the frame's backing quad
    // (thick lines tapered towards the sides). For a perspective vMatrix (projection * view) the
    // w row is the forward axis; an orthographic one has none, so fall back to the view ray.
    float3 view_dir = normalize(cam_pos - world_p);
    float3 forward = float3(vMatrix[0][3], vMatrix[1][3], vMatrix[2][3]);
    bool has_forward = length_squared(forward) > 1e-12;
    float3 facing = has_forward ? normalize(forward) : view_dir;
    
    // Constant width, like a pipe's silhouette: the segments are projected onto the screen plane
    // and the line is extruded perpendicular to what is seen, by the full width. (Extruding by
    // cross(dir, facing) unnormalised thinned the line wherever it turned towards or away from the
    // camera.) A segment pointing straight at the camera projects to a dot and takes the other
    // side's direction.
    float3 dir_in = has_prev ? world_p - world_p_prev : float3(0);
    float3 dir_out = has_next ? world_p_next - world_p : float3(0);
    dir_in -= dot(dir_in, facing) * facing;
    dir_out -= dot(dir_out, facing) * facing;
    bool in_ok = length_squared(dir_in) > 1e-12;
    bool out_ok = length_squared(dir_out) > 1e-12;
    dir_in = in_ok ? normalize(dir_in) : dir_out;
    dir_out = out_ok ? normalize(dir_out) : dir_in;
    
    // Miter join: extrude along the bisector of the incoming and outgoing segments, not just the
    // outgoing one. Extruding perpendicular to the outgoing segment only twists the incoming
    // segment's quad at a corner, so the line pinches to zero width there (a 90 degree corner
    // became a bow-tie).
    float miter_scale = 1.0;
    if (!in_ok && !out_ok) {
        dir = float3(0);                             // a dot on screen: nothing to extrude across
    } else {
        float3 bisector = dir_in + dir_out;
        if (length_squared(bisector) < 1e-6) {
            dir = dir_out;                           // the line doubles back: no sensible miter
        } else {
            dir = normalize(bisector);
            // Both segments keep their full width: the offset grows by 1 / cos(half the turn),
            // capped at 4x. So sharp turns still spike (up to 4x the half width): use pipeline 8
            // (vertex_line3d_round) for round joins.
            miter_scale = 1.0 / max(dot(dir, dir_out), 0.25);
        }
    }
    
    // dir and facing are perpendicular unit vectors, so this is unit length (or 0 for a dot)
    float3 right = cross(dir, facing);
    
    // Scale thickness by depth to maintain constant screen-space pixel width
    float dist = has_forward ? abs(dot(cam_pos - world_p, facing)) : length(cam_pos - world_p);
    float3 offset_p = world_p + right * line_width * miter_scale * dist * side;
    
    // Apply camera projection * view matrix
    vertOut.position = vMatrix * float4(offset_p, 1.0);
    
    // Apply color
    uint c_idx = instanceId * color_strides.x + point_idx * color_strides.y;
    vertOut.color = colors[c_idx];
    
    return vertOut;
}

// Pipeline 8: thick line with round joins and caps (Viewer draws 12N vertices, plain triangles).
// Same inputs and fragment shader as pipeline 4 (vertex_line3d, above).
vertex VertexLineOut vertex_line3d_round(
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

// ---- Text labels (pipeline_state 7, LabelController) ---------------------------------------------
// One Glyph per quad (a letter or a label's background plate), from mesh.glyphs (slot 11). The node's
// world transforms (slot 1) are ordinary transforms: the Viewer draws instances x glyphs, and each
// glyph's anchor goes through its instance's transform, so labels move with their parents.
// The quad is built in clip space around the projected anchor, so labels always face the camera
// and keep a constant size on screen whatever the transform's scale.
struct Glyph {
    packed_float3 anchor;   // the label's bottom centre, in the node's space (packed: the record is 64 bytes)
    float layer;            // 0 plate, 1 letter (letters are pulled a hair nearer in depth)
    float4 rect;            // quad in label units (x0, y0, width, height; 1 unit = one glyph height, y up)
    float4 uv;              // atlas rect (u0, v0, u1, v1)
    float4 rgba;
};

struct LabelParams {
    float size;          // glyph height as a fraction of the viewport height
    float aspect;        // the viewport's width / height
    uint glyph_count;    // glyphs per instance
};

struct LabelOut {
    float4 position [[position]];
    float2 uv;
    float4 color;
};

vertex LabelOut vertex_label(
    uint vertexID [[vertex_id]],
    uint instanceId [[instance_id]],
    constant packed_float3* vertices [[buffer(0)]],     // unit quad corners, 0..1
    constant float4x4* instance_buffer [[buffer(1)]],
    constant float4x4& vMatrix [[buffer(2)]],
    constant LabelParams& params [[buffer(7)]],
    constant Glyph* glyphs [[buffer(11)]]
) {
    Glyph g = glyphs[instanceId % params.glyph_count];
    float4x4 model = instance_buffer[instanceId / params.glyph_count];
    float2 corner = float3(vertices[vertexID]).xy;
    float4 clip = vMatrix * (model * float4(float3(g.anchor), 1.0));
    float2 local = g.rect.xy + corner * g.rect.zw;
    float2 ndc = local * params.size * 2.0;              // NDC spans 2 units of height
    clip.xy += float2(ndc.x / params.aspect, ndc.y) * clip.w;
    clip.z -= g.layer * 2e-4 * clip.w;

    LabelOut out;
    out.position = clip;
    out.uv = float2(mix(g.uv.x, g.uv.z, corner.x), mix(g.uv.w, g.uv.y, corner.y));   // atlas v runs down
    out.color = g.rgba;
    return out;
}

fragment float4 fragment_label(LabelOut in [[stage_in]],
                               texture2d<float, access::sample> atlas [[texture(0)]]) {
    constexpr sampler s(address::clamp_to_edge, filter::linear);
    float a = atlas.sample(s, in.uv).a * in.color.a;
    if (a < 0.02) discard_fragment();
    return float4(in.color.rgb, a);
}

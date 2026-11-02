#pragma once
#include "../matrix/matrix.h"

#include "GeoNode.cpp"

namespace MeshPrimitives {

    // A host [rows, cols] Float matrix, refcounted, to fill in place.
    inline matrix host_floats(size_m rows, size_m cols) {
        matrix m = matrix::withShape({rows, cols}, dtype::Float);
        m.begin_refcount();
        return m;
    }

    // [count, 3]: the same normal on every vertex (flat shapes).
    inline matrix constant_normals(size_m count, simd_float3 n) {
        matrix m = host_floats(count, 3);
        float* p = static_cast<float*>(m.buffer);
        for (size_m i = 0; i < count; i++) p[3 * i] = n.x, p[3 * i + 1] = n.y, p[3 * i + 2] = n.z;
        return m;
    }

    inline Mesh triangle(float base = 1.0f, float height = 1.0f) {
        float hw = base / 2.0f;
        float hh = height / 2.0f;
        
        Mesh m;
        m.vert_position = matrix::of<float>({
            {-hw, -hh, 0.0f},
            { hw, -hh, 0.0f},
            { 0.0f, hh, 0.0f}
        });
        m.uv_coords = matrix::of<float>({
            {0.0f, 0.0f},
            {1.0f, 0.0f},
            {0.5f, 1.0f}
        });
        m.indices = matrix::of<uint32_t>({
            0, 1, 2
        });
        m.normals = constant_normals(3, simd_make_float3(0.0f, 0.0f, 1.0f));
        m.vert_position.begin_refcount();
        m.indices.begin_refcount();
        m.uv_coords.begin_refcount();

        return m;
    }

    inline Mesh quad(float width = 1.0f, float height = 1.0f) {
        float hw = width / 2.0f;
        float hh = height / 2.0f;
        
        Mesh m;
        m.vert_position = matrix::of<float>({
            {-hw, -hh, 0.0f}, // 0: bottom-left
            { hw, -hh, 0.0f}, // 1: bottom-right
            { hw,  hh, 0.0f}, // 2: top-right
            {-hw,  hh, 0.0f}  // 3: top-left
        });
        m.uv_coords = matrix::of<float>({
            {0.0f, 0.0f},
            {1.0f, 0.0f},
            {1.0f, 1.0f},
            {0.0f, 1.0f}
        });
        m.indices = matrix::of<uint32_t>({
            0, 1, 2,
            0, 2, 3
        });
        m.normals = constant_normals(4, simd_make_float3(0.0f, 0.0f, 1.0f));
        m.vert_position.begin_refcount();
        m.indices.begin_refcount();
        m.uv_coords.begin_refcount();

        return m;
    }

    inline matrix quad_edges() {
        auto edges = matrix::of<uint32_t>({
            {0, 1},
            {1, 2},
            {2, 3},
            {3, 0}
        });
        edges.begin_refcount();
        return edges;
    }

    inline matrix cube_edges() {
        auto edges = matrix::of<uint32_t>({
            // Bottom face
            {0, 1},  {1, 2},  {2, 3},  {3, 0},
            // Top face
            {4, 5},  {5, 6},  {6, 7},  {7, 4},
            // Pillars
            {0, 4},  {1, 5},  {2, 6},  {3, 7}
        });
        edges.begin_refcount();
        return edges;
    }

    // 24 vertices, 4 per face, so each face has its own normal (shared corners would shade like a blob).
    // Vertices 0-7 are the 8 corners in their original order (front face 0-3, back face 4-7), so
    // cube_edges() still indexes the corners.
    inline Mesh cube(float width = 1.0f, float height = 1.0f, float depth = 1.0f) {
        float hw = width / 2.0f;
        float hh = height / 2.0f;
        float hd = depth / 2.0f;

        Mesh m;
        m.vert_position = matrix::of<float>({
            {-hw, -hh,  hd}, { hw, -hh,  hd}, { hw,  hh,  hd}, {-hw,  hh,  hd},   // 0-3   front  (+z)
            {-hw, -hh, -hd}, { hw, -hh, -hd}, { hw,  hh, -hd}, {-hw,  hh, -hd},   // 4-7   back   (-z)
            { hw, -hh,  hd}, { hw, -hh, -hd}, { hw,  hh, -hd}, { hw,  hh,  hd},   // 8-11  right  (+x)
            {-hw, -hh, -hd}, {-hw, -hh,  hd}, {-hw,  hh,  hd}, {-hw,  hh, -hd},   // 12-15 left   (-x)
            {-hw,  hh,  hd}, { hw,  hh,  hd}, { hw,  hh, -hd}, {-hw,  hh, -hd},   // 16-19 top    (+y)
            {-hw, -hh, -hd}, { hw, -hh, -hd}, { hw, -hh,  hd}, {-hw, -hh,  hd},   // 20-23 bottom (-y)
        });
        m.normals = matrix::of<float>({
            {0, 0, 1}, {0, 0, 1}, {0, 0, 1}, {0, 0, 1},
            {0, 0, -1}, {0, 0, -1}, {0, 0, -1}, {0, 0, -1},
            {1, 0, 0}, {1, 0, 0}, {1, 0, 0}, {1, 0, 0},
            {-1, 0, 0}, {-1, 0, 0}, {-1, 0, 0}, {-1, 0, 0},
            {0, 1, 0}, {0, 1, 0}, {0, 1, 0}, {0, 1, 0},
            {0, -1, 0}, {0, -1, 0}, {0, -1, 0}, {0, -1, 0},
        });
        m.uv_coords = matrix::of<float>({
            {0, 0}, {1, 0}, {1, 1}, {0, 1},  {0, 0}, {1, 0}, {1, 1}, {0, 1},
            {0, 0}, {1, 0}, {1, 1}, {0, 1},  {0, 0}, {1, 0}, {1, 1}, {0, 1},
            {0, 0}, {1, 0}, {1, 1}, {0, 1},  {0, 0}, {1, 0}, {1, 1}, {0, 1},
        });
        m.indices = matrix::of<uint32_t>({
             0,  1,  2,   0,  2,  3, // Front
             8,  9, 10,   8, 10, 11, // Right
             5,  4,  7,   5,  7,  6, // Back
            12, 13, 14,  12, 14, 15, // Left
            16, 17, 18,  16, 18, 19, // Top
            20, 21, 22,  20, 22, 23  // Bottom
        });
        m.vert_position.begin_refcount();
        m.normals.begin_refcount();
        m.indices.begin_refcount();
        m.uv_coords.begin_refcount();

        return m;
    }

    inline Mesh circle(float radius = 1.0f, int subdivisions = 32) {
        Mesh m;
        m.vert_position = matrix(2, dtype::Float);
        m.vert_position.shape()[0] = subdivisions + 1;
        m.vert_position.shape()[1] = 3;
        m.vert_position.total_size = (subdivisions + 1) * 3;
        m.vert_position.calcStrides();
        m.vert_position.buffer = new uint8_t[m.vert_position.total_size * sizeof(float)];
        
        m.indices = matrix(1, dtype::UInt32);
        m.indices.shape()[0] = 3 * subdivisions;
        m.indices.total_size = 3 * subdivisions;
        m.indices.calcStrides();
        m.indices.buffer = new uint8_t[m.indices.total_size * sizeof(uint32_t)];
        
        m.vert_position.at<float>(0, 0) = 0.0f;
        m.vert_position.at<float>(0, 1) = 0.0f;
        m.vert_position.at<float>(0, 2) = 0.0f;
        
        m.uv_coords = matrix(2, dtype::Float);
        m.uv_coords.shape()[0] = subdivisions + 1;
        m.uv_coords.shape()[1] = 2;
        m.uv_coords.total_size = (subdivisions + 1) * 2;
        m.uv_coords.calcStrides();
        m.uv_coords.buffer = new uint8_t[m.uv_coords.total_size * sizeof(float)];
        
        m.uv_coords.at<float>(0, 0) = 0.5f;
        m.uv_coords.at<float>(0, 1) = 0.5f;
        
        matrix vertView(2, dtype::Float);
        m.vert_position.shareBuffer(vertView);
        vertView.buffer = (uint8_t*)vertView.buffer + 3 * sizeof(float);
        vertView.shape()[0] = subdivisions;
        vertView.shape()[1] = 3;
        vertView.calcStrides();
        vertView.total_size = vertView.accumul(0, 2);
        
        matrix uvView(2, dtype::Float);
        m.uv_coords.shareBuffer(uvView);
        uvView.buffer = (uint8_t*)uvView.buffer + 2 * sizeof(float);
        uvView.shape()[0] = subdivisions;
        uvView.shape()[1] = 2;
        uvView.calcStrides();
        uvView.total_size = uvView.accumul(0, 2);
        
        float stride = 2.0f * M_PI / subdivisions;
        for (int i = 0; i < subdivisions; i++) {
            float theta = i * stride;
            vertView.at<float>(i, 0) = radius * cos(theta);
            vertView.at<float>(i, 1) = radius * sin(theta);
            vertView.at<float>(i, 2) = 0.0f;
            
            uvView.at<float>(i, 0) = 0.5f + 0.5f * cos(theta);
            uvView.at<float>(i, 1) = 0.5f + 0.5f * sin(theta);
        }
        
        matrix indicesView(2, dtype::UInt32);
        m.indices.shareBuffer(indicesView);
        indicesView.shape()[0] = subdivisions;
        indicesView.shape()[1] = 3;
        indicesView.calcStrides();
        indicesView.total_size = indicesView.accumul(0, 2);
        
        for (int i = 0; i < subdivisions - 1; i++) {
            indicesView.at<uint32_t>(i, 0) = 0;
            indicesView.at<uint32_t>(i, 1) = 1 + i;
            indicesView.at<uint32_t>(i, 2) = 1 + i + 1;
        }
        indicesView.at<uint32_t>(subdivisions - 1, 0) = 0;
        indicesView.at<uint32_t>(subdivisions - 1, 1) = 1 + subdivisions - 1;
        indicesView.at<uint32_t>(subdivisions - 1, 2) = 1 + 0;
        m.normals = constant_normals(subdivisions + 1, simd_make_float3(0.0f, 0.0f, 1.0f));
        m.vert_position.begin_refcount();
        m.indices.begin_refcount();
        m.uv_coords.begin_refcount();

        return m;
    }

    inline Mesh sphere(float radius = 1.0f, int radialSubdivisions = 16, int linearSubdivisions = 16) {
        Mesh m;
        int vertexCount = radialSubdivisions * linearSubdivisions + 2;
        int indexCount = radialSubdivisions * 3 + 6 * radialSubdivisions * (linearSubdivisions - 1) + radialSubdivisions * 3;
        
        m.vert_position = matrix(2, dtype::Float);
        m.vert_position.shape()[0] = vertexCount;
        m.vert_position.shape()[1] = 3;
        m.vert_position.total_size = vertexCount * 3;
        m.vert_position.calcStrides();
        m.vert_position.buffer = new uint8_t[m.vert_position.total_size * sizeof(float)];
        
        m.indices = matrix(1, dtype::UInt32);
        m.indices.shape()[0] = indexCount;
        m.indices.total_size = indexCount;
        m.indices.calcStrides();
        m.indices.buffer = new uint8_t[m.indices.total_size * sizeof(uint32_t)];
        
        // Poles
        m.vert_position.at<float>(0, 0) = 0.0f;
        m.vert_position.at<float>(0, 1) = radius;
        m.vert_position.at<float>(0, 2) = 0.0f;
        
        m.vert_position.at<float>(vertexCount - 1, 0) = 0.0f;
        m.vert_position.at<float>(vertexCount - 1, 1) = -radius;
        m.vert_position.at<float>(vertexCount - 1, 2) = 0.0f;
        
        m.uv_coords = matrix(2, dtype::Float);
        m.uv_coords.shape()[0] = vertexCount;
        m.uv_coords.shape()[1] = 2;
        m.uv_coords.total_size = vertexCount * 2;
        m.uv_coords.calcStrides();
        m.uv_coords.buffer = new uint8_t[m.uv_coords.total_size * sizeof(float)];
        
        // Poles
        m.uv_coords.at<float>(0, 0) = 0.5f;
        m.uv_coords.at<float>(0, 1) = 0.0f; // Top pole (v = 0)
        
        m.uv_coords.at<float>(vertexCount - 1, 0) = 0.5f;
        m.uv_coords.at<float>(vertexCount - 1, 1) = 1.0f; // Bottom pole (v = 1)
        
        matrix vertView(3, dtype::Float);
        m.vert_position.shareBuffer(vertView);
        vertView.buffer = (uint8_t*)vertView.buffer + 3 * sizeof(float); // Skip top pole
        vertView.shape()[0] = linearSubdivisions;
        vertView.shape()[1] = radialSubdivisions;
        vertView.shape()[2] = 3;
        vertView.calcStrides();
        vertView.total_size = vertView.accumul(0, 3);
        
        matrix uvView(3, dtype::Float);
        m.uv_coords.shareBuffer(uvView);
        uvView.buffer = (uint8_t*)uvView.buffer + 2 * sizeof(float); // Skip top pole
        uvView.shape()[0] = linearSubdivisions;
        uvView.shape()[1] = radialSubdivisions;
        uvView.shape()[2] = 2;
        uvView.calcStrides();
        uvView.total_size = uvView.accumul(0, 3);
        
        float thetaStride = 2.0f * M_PI / radialSubdivisions;
        float polarAngle = M_PI / (linearSubdivisions + 1);
        
        for (int h = 0; h < linearSubdivisions; h++) {
            float rMag = radius * sin(polarAngle);
            float yV = radius * cos(polarAngle);
            float vCoord = polarAngle / M_PI; // 0 to 1
            for (int i = 0; i < radialSubdivisions; i++) {
                float theta = i * thetaStride;
                float uCoord = (float)i / radialSubdivisions; // 0 to 1
                
                vertView.at<float>(h, i, 0) = rMag * cos(theta);
                vertView.at<float>(h, i, 1) = yV;
                vertView.at<float>(h, i, 2) = rMag * sin(theta);
                
                uvView.at<float>(h, i, 0) = uCoord;
                uvView.at<float>(h, i, 1) = vCoord;
            }
            polarAngle += M_PI / (linearSubdivisions + 1);
        }
        
        matrix topCapView(2, dtype::UInt32);
        m.indices.shareBuffer(topCapView);
        topCapView.shape()[0] = radialSubdivisions;
        topCapView.shape()[1] = 3;
        topCapView.calcStrides();
        topCapView.total_size = topCapView.accumul(0, 2);
        
        for (int i = 0; i < radialSubdivisions; i++) {
            topCapView.at<uint32_t>(i, 0) = 0;
            topCapView.at<uint32_t>(i, 1) = 1 + i;
            topCapView.at<uint32_t>(i, 2) = 1 + (i + 1) % radialSubdivisions;
        }
        
        matrix middleQuadsView(3, dtype::UInt32);
        m.indices.shareBuffer(middleQuadsView);
        middleQuadsView.buffer = (uint8_t*)middleQuadsView.buffer + (3 * radialSubdivisions) * sizeof(uint32_t);
        middleQuadsView.shape()[0] = linearSubdivisions - 1;
        middleQuadsView.shape()[1] = radialSubdivisions;
        middleQuadsView.shape()[2] = 6;
        middleQuadsView.calcStrides();
        middleQuadsView.total_size = middleQuadsView.accumul(0, 3);
        
        for (int j = 0; j < linearSubdivisions - 1; j++) {
            for (int i = 0; i < radialSubdivisions; i++) {
                int nextI = (i + 1) % radialSubdivisions;
                int currRow = 1 + j * radialSubdivisions;
                int nextRow = 1 + (j + 1) * radialSubdivisions;
                
                middleQuadsView.at<uint32_t>(j, i, 0) = currRow + i;
                middleQuadsView.at<uint32_t>(j, i, 1) = nextRow + i;
                middleQuadsView.at<uint32_t>(j, i, 2) = nextRow + nextI;
                
                middleQuadsView.at<uint32_t>(j, i, 3) = currRow + i;
                middleQuadsView.at<uint32_t>(j, i, 4) = currRow + nextI;
                middleQuadsView.at<uint32_t>(j, i, 5) = nextRow + nextI;
            }
        }
        
        matrix bottomCapView(2, dtype::UInt32);
        m.indices.shareBuffer(bottomCapView);
        bottomCapView.buffer = (uint8_t*)middleQuadsView.buffer + middleQuadsView.total_size * sizeof(uint32_t);
        bottomCapView.shape()[0] = radialSubdivisions;
        bottomCapView.shape()[1] = 3;
        bottomCapView.calcStrides();
        bottomCapView.total_size = bottomCapView.accumul(0, 2);
        
        for (int i = 0; i < radialSubdivisions; i++) {
            bottomCapView.at<uint32_t>(i, 0) = vertexCount - 1;
            bottomCapView.at<uint32_t>(i, 1) = 1 + (linearSubdivisions - 1) * radialSubdivisions + i;
            bottomCapView.at<uint32_t>(i, 2) = 1 + (linearSubdivisions - 1) * radialSubdivisions + (i + 1) % radialSubdivisions;
        }

        // On a sphere the normal is the position over the radius.
        m.normals = host_floats(vertexCount, 3);
        {
            const float* pos = static_cast<const float*>(m.vert_position.buffer);
            float* nrm = static_cast<float*>(m.normals.buffer);
            for (int v = 0; v < 3 * vertexCount; v++) nrm[v] = pos[v] / radius;
        }
        m.vert_position.begin_refcount();
        m.indices.begin_refcount();
        m.uv_coords.begin_refcount();

        return m;
    }

    // Caps and side have separate vertices so the rim is a hard edge: the caps face straight down / up,
    // the side faces outward. Vertices: 0 bottom centre, 1..n bottom cap ring, n+1..2n top cap ring,
    // 2n+1 top centre, then the side's own rings: 2n+2.. bottom, 3n+2.. top.
    inline Mesh cylinder(float radius = 1.0f, float length = 1.0f, int subdivisions = 32) {
        const int n = subdivisions;
        const int vertexCount = 4 * n + 2;
        const int topCentre = 2 * n + 1, sideBottom = 2 * n + 2, sideTop = 3 * n + 2;

        Mesh m;
        m.vert_position = host_floats(vertexCount, 3);
        m.normals = host_floats(vertexCount, 3);
        m.uv_coords = host_floats(vertexCount, 2);
        m.indices = matrix::withShape({(size_m)(12 * n)}, dtype::UInt32);   // 3 bottom + 6 side + 3 top per segment
        m.indices.begin_refcount();

        float* P = static_cast<float*>(m.vert_position.buffer);
        float* N = static_cast<float*>(m.normals.buffer);
        float* U = static_cast<float*>(m.uv_coords.buffer);
        auto put = [&](int v, simd_float3 p, simd_float3 nrm, simd_float2 uv) {
            P[3 * v] = p.x, P[3 * v + 1] = p.y, P[3 * v + 2] = p.z;
            N[3 * v] = nrm.x, N[3 * v + 1] = nrm.y, N[3 * v + 2] = nrm.z;
            U[2 * v] = uv.x, U[2 * v + 1] = uv.y;
        };

        const simd_float3 down = {0, -1, 0}, up = {0, 1, 0};
        put(0, simd_make_float3(0, 0, 0), down, simd_make_float2(0.5f, 0.5f));
        put(topCentre, simd_make_float3(0, length, 0), up, simd_make_float2(0.5f, 0.5f));
        const float stride = 2.0f * M_PI / n;
        for (int i = 0; i < n; i++) {
            const float c = cos(i * stride), s = sin(i * stride);
            const simd_float2 cap_uv = simd_make_float2(0.5f + 0.5f * c, 0.5f + 0.5f * s);
            const float u = (float)i / n;
            put(1 + i,          simd_make_float3(radius * c, 0, radius * s),      down,                      cap_uv);
            put(1 + n + i,      simd_make_float3(radius * c, length, radius * s), up,                        cap_uv);
            put(sideBottom + i, simd_make_float3(radius * c, 0, radius * s),      simd_make_float3(c, 0, s), simd_make_float2(u, 0));
            put(sideTop + i,    simd_make_float3(radius * c, length, radius * s), simd_make_float3(c, 0, s), simd_make_float2(u, 1));
        }

        uint32_t* idx_buf = (uint32_t*)m.indices.buffer;
        int idxOffset = 0;
        for (int i = 0; i < n; i++) {
            int next_i = (i + 1) % n;

            // Bottom cap triangle
            idx_buf[idxOffset++] = 0;
            idx_buf[idxOffset++] = 1 + i;
            idx_buf[idxOffset++] = 1 + next_i;

            // Side quad (2 triangles), on the side's own rings
            idx_buf[idxOffset++] = sideBottom + i;
            idx_buf[idxOffset++] = sideTop + i;
            idx_buf[idxOffset++] = sideTop + next_i;

            idx_buf[idxOffset++] = sideBottom + i;
            idx_buf[idxOffset++] = sideTop + next_i;
            idx_buf[idxOffset++] = sideBottom + next_i;

            // Top cap triangle
            idx_buf[idxOffset++] = topCentre;
            idx_buf[idxOffset++] = 1 + n + i;
            idx_buf[idxOffset++] = 1 + n + next_i;
        }

        return m;
    }

    // Base and side have separate vertices (a hard rim), and the tip has one vertex per segment, each with
    // that segment's normal, so the point shades correctly. Vertices: 0 base centre, 1..n base ring,
    // n+1 the tip (kept at its old index), then n+2.. the side ring and 2n+2.. the per-segment tips.
    inline Mesh cone(float radius = 1.0f, float length = 1.0f, int subdivisions = 32) {
        const int n = subdivisions;
        const int vertexCount = 3 * n + 2;
        const int tip = n + 1, sideRing = n + 2, tips = 2 * n + 2;

        Mesh m;
        m.vert_position = host_floats(vertexCount, 3);
        m.normals = host_floats(vertexCount, 3);
        m.uv_coords = host_floats(vertexCount, 2);
        m.indices = matrix::withShape({(size_m)(6 * n)}, dtype::UInt32);   // 3 base + 3 side per segment
        m.indices.begin_refcount();

        float* P = static_cast<float*>(m.vert_position.buffer);
        float* N = static_cast<float*>(m.normals.buffer);
        float* U = static_cast<float*>(m.uv_coords.buffer);
        auto put = [&](int v, simd_float3 p, simd_float3 nrm, simd_float2 uv) {
            P[3 * v] = p.x, P[3 * v + 1] = p.y, P[3 * v + 2] = p.z;
            N[3 * v] = nrm.x, N[3 * v + 1] = nrm.y, N[3 * v + 2] = nrm.z;
            U[2 * v] = uv.x, U[2 * v + 1] = uv.y;
        };
        // Outward normal of the slanted side at angle theta: perpendicular to the slope (radius over length).
        auto side_normal = [&](float theta) {
            return simd_normalize(simd_make_float3(length * cos(theta), radius, length * sin(theta)));
        };

        const simd_float3 down = {0, -1, 0};
        put(0, simd_make_float3(0, 0, 0), down, simd_make_float2(0.5f, 0.5f));
        put(tip, simd_make_float3(0, length, 0), simd_make_float3(0, 1, 0), simd_make_float2(0.5f, 0.5f));
        const float stride = 2.0f * M_PI / n;
        for (int i = 0; i < n; i++) {
            const float theta = i * stride, c = cos(theta), s = sin(theta);
            put(1 + i,        simd_make_float3(radius * c, 0, radius * s), down,                            simd_make_float2(0.5f + 0.5f * c, 0.5f + 0.5f * s));
            put(sideRing + i, simd_make_float3(radius * c, 0, radius * s), side_normal(theta),              simd_make_float2((float)i / n, 0));
            put(tips + i,     simd_make_float3(0, length, 0),              side_normal(theta + 0.5f * stride), simd_make_float2((i + 0.5f) / n, 1));
        }

        uint32_t* idx_buf = (uint32_t*)m.indices.buffer;
        int idxOffset = 0;
        for (int i = 0; i < n; i++) {
            int next_i = (i + 1) % n;

            // Base triangle
            idx_buf[idxOffset++] = 0;
            idx_buf[idxOffset++] = 1 + i;
            idx_buf[idxOffset++] = 1 + next_i;

            // Side triangle, with this segment's own tip vertex
            idx_buf[idxOffset++] = sideRing + i;
            idx_buf[idxOffset++] = tips + i;
            idx_buf[idxOffset++] = sideRing + next_i;
        }

        return m;
    }


}

#pragma once
#include "matrix.h"

#include "GeoNode.cpp"

namespace MeshPrimitives {

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
        m.vert_position.begin_refcount();
        m.indices.begin_refcount();
        m.uv_coords.begin_refcount();

        return m;
    }

    inline matrix quad_edges() {
        auto edges = matrix::of<uint32_t>({
            0, 1,
            1, 2,
            2, 3,
            3, 0
        }).reshape(4, 2);
        edges.begin_refcount();
        return edges;
    }

    inline matrix cube_edges() {
        auto edges = matrix::of<uint32_t>({
            // Bottom face
            0, 1,  1, 2,  2, 3,  3, 0,
            // Top face
            4, 5,  5, 6,  6, 7,  7, 4,
            // Pillars
            0, 4,  1, 5,  2, 6,  3, 7
        }).reshape(12, 2);
        edges.begin_refcount();
        return edges;
    }

    inline Mesh cube(float width = 1.0f, float height = 1.0f, float depth = 1.0f) {
        float hw = width / 2.0f;
        float hh = height / 2.0f;
        float hd = depth / 2.0f;
        
        Mesh m;
        m.vert_position = matrix::of<float>({
            {-hw, -hh,  hd}, // 0: front-bottom-left
            { hw, -hh,  hd}, // 1: front-bottom-right
            { hw,  hh,  hd}, // 2: front-top-right
            {-hw,  hh,  hd}, // 3: front-top-left
            {-hw, -hh, -hd}, // 4: back-bottom-left
            { hw, -hh, -hd}, // 5: back-bottom-right
            { hw,  hh, -hd}, // 6: back-top-right
            {-hw,  hh, -hd}  // 7: back-top-left
        });
        m.uv_coords = matrix::of<float>({
            {0.0f, 0.0f}, // 0
            {1.0f, 0.0f}, // 1
            {1.0f, 1.0f}, // 2
            {0.0f, 1.0f}, // 3
            {0.0f, 0.0f}, // 4
            {1.0f, 0.0f}, // 5
            {1.0f, 1.0f}, // 6
            {0.0f, 1.0f}  // 7
        });
        m.indices = matrix::of<uint32_t>({
            0, 1, 2,  0, 2, 3, // Front
            1, 5, 6,  1, 6, 2, // Right
            5, 4, 7,  5, 7, 6, // Back
            4, 0, 3,  4, 3, 7, // Left
            3, 2, 6,  3, 6, 7, // Top
            4, 5, 1,  4, 1, 0  // Bottom
        });
        m.vert_position.begin_refcount();
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
        m.vert_position.begin_refcount();
        m.indices.begin_refcount();
        m.uv_coords.begin_refcount();

        return m;
    }

    inline Mesh cylinder(float radius = 1.0f, float length = 1.0f, int subdivisions = 32) {
        Mesh m;
        int vertexCount = subdivisions * 2 + 2;
        int indexCount = subdivisions * 12; // 3 (bottom) + 6 (side) + 3 (top) per subdivision
        
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
        
        m.uv_coords = matrix(2, dtype::Float);
        m.uv_coords.shape()[0] = vertexCount;
        m.uv_coords.shape()[1] = 2;
        m.uv_coords.total_size = vertexCount * 2;
        m.uv_coords.calcStrides();
        m.uv_coords.buffer = new uint8_t[m.uv_coords.total_size * sizeof(float)];

        // bottom center
        m.vert_position.at<float>(0, 0) = 0.0f;
        m.vert_position.at<float>(0, 1) = 0.0f;
        m.vert_position.at<float>(0, 2) = 0.0f;
        
        m.uv_coords.at<float>(0, 0) = 0.5f;
        m.uv_coords.at<float>(0, 1) = 0.5f;
        
        // top center
        m.vert_position.at<float>(vertexCount - 1, 0) = 0.0f;
        m.vert_position.at<float>(vertexCount - 1, 1) = length;
        m.vert_position.at<float>(vertexCount - 1, 2) = 0.0f;

        m.uv_coords.at<float>(vertexCount - 1, 0) = 0.5f;
        m.uv_coords.at<float>(vertexCount - 1, 1) = 0.5f;

        float stride = 2.0f * M_PI / subdivisions;
        
        for (int i = 0; i < subdivisions; i++) {
            float theta = i * stride;
            float c = cos(theta);
            float s = sin(theta);
            
            // Bottom ring
            m.vert_position.at<float>(1 + i, 0) = radius * c;
            m.vert_position.at<float>(1 + i, 1) = 0.0f;
            m.vert_position.at<float>(1 + i, 2) = radius * s;
            
            m.uv_coords.at<float>(1 + i, 0) = 0.5f + 0.5f * c;
            m.uv_coords.at<float>(1 + i, 1) = 0.5f + 0.5f * s;
            
            // Top ring
            m.vert_position.at<float>(1 + subdivisions + i, 0) = radius * c;
            m.vert_position.at<float>(1 + subdivisions + i, 1) = length;
            m.vert_position.at<float>(1 + subdivisions + i, 2) = radius * s;

            m.uv_coords.at<float>(1 + subdivisions + i, 0) = 0.5f + 0.5f * c;
            m.uv_coords.at<float>(1 + subdivisions + i, 1) = 0.5f + 0.5f * s;
        }
        
        uint32_t* idx_buf = (uint32_t*)m.indices.buffer;
        int idxOffset = 0;
        for (int i = 0; i < subdivisions; i++) {
            int next_i = (i + 1) % subdivisions;
            
            // Bottom cap triangle
            idx_buf[idxOffset++] = 0;
            idx_buf[idxOffset++] = 1 + i;
            idx_buf[idxOffset++] = 1 + next_i;
            
            // Side quad (2 triangles)
            idx_buf[idxOffset++] = 1 + i;
            idx_buf[idxOffset++] = 1 + subdivisions + i;
            idx_buf[idxOffset++] = 1 + subdivisions + next_i;
            
            idx_buf[idxOffset++] = 1 + i;
            idx_buf[idxOffset++] = 1 + subdivisions + next_i;
            idx_buf[idxOffset++] = 1 + next_i;
            
            // Top cap triangle
            idx_buf[idxOffset++] = vertexCount - 1;
            idx_buf[idxOffset++] = 1 + subdivisions + i;
            idx_buf[idxOffset++] = 1 + subdivisions + next_i;
        }
        m.vert_position.begin_refcount();
        m.indices.begin_refcount();
        m.uv_coords.begin_refcount();

        return m;
    }

    inline Mesh cone(float radius = 1.0f, float length = 1.0f, int subdivisions = 32) {
        Mesh m;
        int vertexCount = subdivisions + 2;
        int indexCount = subdivisions * 6; // 3 (bottom) + 3 (side) per subdivision
        
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
        
        m.uv_coords = matrix(2, dtype::Float);
        m.uv_coords.shape()[0] = vertexCount;
        m.uv_coords.shape()[1] = 2;
        m.uv_coords.total_size = vertexCount * 2;
        m.uv_coords.calcStrides();
        m.uv_coords.buffer = new uint8_t[m.uv_coords.total_size * sizeof(float)];

        // bottom center
        m.vert_position.at<float>(0, 0) = 0.0f;
        m.vert_position.at<float>(0, 1) = 0.0f;
        m.vert_position.at<float>(0, 2) = 0.0f;
        
        m.uv_coords.at<float>(0, 0) = 0.5f;
        m.uv_coords.at<float>(0, 1) = 0.5f;
        
        // top tip
        m.vert_position.at<float>(vertexCount - 1, 0) = 0.0f;
        m.vert_position.at<float>(vertexCount - 1, 1) = length;
        m.vert_position.at<float>(vertexCount - 1, 2) = 0.0f;

        m.uv_coords.at<float>(vertexCount - 1, 0) = 0.5f;
        m.uv_coords.at<float>(vertexCount - 1, 1) = 0.5f;

        float stride = 2.0f * M_PI / subdivisions;
        
        for (int i = 0; i < subdivisions; i++) {
            float theta = i * stride;
            float c = cos(theta);
            float s = sin(theta);
            
            // Bottom ring
            m.vert_position.at<float>(1 + i, 0) = radius * c;
            m.vert_position.at<float>(1 + i, 1) = 0.0f;
            m.vert_position.at<float>(1 + i, 2) = radius * s;
            
            m.uv_coords.at<float>(1 + i, 0) = 0.5f + 0.5f * c;
            m.uv_coords.at<float>(1 + i, 1) = 0.5f + 0.5f * s;
        }
        
        uint32_t* idx_buf = (uint32_t*)m.indices.buffer;
        int idxOffset = 0;
        for (int i = 0; i < subdivisions; i++) {
            int next_i = (i + 1) % subdivisions;
            
            // Bottom cap triangle
            idx_buf[idxOffset++] = 0;
            idx_buf[idxOffset++] = 1 + i;
            idx_buf[idxOffset++] = 1 + next_i;
            
            // Side triangle
            idx_buf[idxOffset++] = 1 + i;
            idx_buf[idxOffset++] = vertexCount - 1;
            idx_buf[idxOffset++] = 1 + next_i;
        }
        m.vert_position.begin_refcount();
        m.indices.begin_refcount();
        m.uv_coords.begin_refcount();

        return m;
    }

}

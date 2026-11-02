//
//  Shaders.metal
//  AdityaIntelligenceProMax
//
//  Created by Manoj Kumar on 10/03/25.
//

#include <metal_stdlib>
#include <metal_simdgroup_matrix>
using namespace metal;

typedef struct
{
    float2 position [[attribute(0)]];
    float2 texCoord [[attribute(1)]];
} Vertex;

typedef struct
{
    float3 position [[attribute(0)]];
    float4 colour [[attribute(1)]];
} Point3D;

typedef struct
{
    float4 position [[position]];
    float2 texCoord;
} ColorInOut;

typedef struct
{
    float4 position [[position]];
    float4 worldSpacePosition;
} InfiniteGridColorInOut;

// Display a 2D texture.
vertex ColorInOut planeVertexShader(Vertex in [[stage_in]])
{
    ColorInOut out;
    out.position = float4(in.position, 0.99999f, 1.0f);
    out.texCoord = in.texCoord;
    return out;
}

// Shade a 2D plane by passing through the texture inputs.
fragment float4 planeFragmentShader(ColorInOut in [[stage_in]], texture2d<float, access::sample> textureIn [[ texture(0) ]])
{
    constexpr sampler colorSampler(address::clamp_to_edge, filter::linear);
    float4 sample = textureIn.sample(colorSampler, in.texCoord);
    return sample;
}

constant float2 verticies[4] = {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
constant float2 textCoords[4] = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
constant float gridSize = 0.05f;
constant float lineWidth = 0.001f;
constant float4 Thickcolour = {1, 1, 1, 1};
constant float3 xAxisColor = float3(1.0, 0.3, 0.3); // Red for X axis
constant float3 zAxisColor = float3(0.3, 0.3, 1.0); // Blue for Z axis

template<typename T, typename U>
constexpr T mod(T x, U y) { return x - y * floor( x / y ); }


vertex InfiniteGridColorInOut InfiniteGridVertexShader(uint vertextID [[vertex_id]], constant float4x4& cam [[buffer(0)]], constant float3& worldCamPosition [[buffer(1)]], constant float4x4& inverseModelViewProjectionMatrix [[buffer(2)]])
{
    InfiniteGridColorInOut out;
//    float3 pos = float3(verticies[vertextID].x, 0.0f, verticies[vertextID].y);
//    pos.x += worldCamPosition.x;
//    pos.y += worldCamPosition.y;
//    pos.z += worldCamPosition.z;
    
//    out.position =  cam * float4(verticies[vertextID].x, 0.0f, verticies[vertextID].y, 1.0f) - float4(cam.columns[3].xyz, 0.0f) + float4(0, 0, 0.5, 0);
//    out.position = cam * float4(pos, 1.0f)  + float4(0, 0, 0.7, 0);
//    out.worldSpacePosition = float4(verticies[vertextID].x, 0.0f, verticies[vertextID].y, 1.0f);
    float4 pos = float4(40 * verticies[vertextID].x, 0.0f, 40 * verticies[vertextID].y, 1.0f);
    pos.x += worldCamPosition.x;
    pos.z += worldCamPosition.z;
    
    float4 tex = float4( verticies[vertextID].x, 0.0f, verticies[vertextID].y, 1.0f);
    
    out.worldSpacePosition = float4(verticies[vertextID].x + worldCamPosition.x / 40, verticies[vertextID].y + worldCamPosition.z / 40, 0, 0);
    out.position = cam * pos;
    
    
    return out;
}

int diracBelta(float value) {
    if (value < lineWidth || (gridSize - lineWidth) < value) {
        return 1;
    } else {
        return 0;
    }
}
inline float max(float2 v) {
    return max(v.x, v.y);
}

// Shade a 2D plane by passing through the texture inputs.
fragment float4 InfiniteGridFragmentShader(InfiniteGridColorInOut in [[stage_in]], constant float3& cameraPosition [[buffer(0)]], constant float4x4& inverseModelViewProjectionMatrix [[buffer(1)]])
{
    
//    float gGridCellSize = 0.025;
//    float4 gGridColorThin = float4(0.5, 0.5, 0.5, 1.0);
//    float4 gGridColorThick = float4(0.0, 0.0, 0.0, 1.0);
//    float Lodoa = fmod(in.worldSpacePosition.z+1, gGridCellSize);
//    float4 Color;
//    Color = gGridColorThick;
//    Color.a *= Lodoa / 0.025;
//    float4 Temp = (Color * Color.a) + (1 - Color.a) * float4(1,1,1,1);
//    Temp.a = 1;
//    auto color = float4(mod(in.worldSpacePosition.x, gridSize), mod(in.worldSpacePosition.y, gridSize), 0, 1);
//    auto color = float4(1, 1, 1, mod(in.worldSpacePosition.x, gridSize) * mod(in.worldSpacePosition.y, gridSize));
//    auto color = float4(1, 1, 1, diracBelta( mod(in.worldSpacePosition.x, gridSize) ) | diracBelta( mod(in.worldSpacePosition.y, gridSize) ));
    float zAxis = abs(in.worldSpacePosition.y) < 0.01 ? 1.0 : 0.0;
    float xAxis = abs(in.worldSpacePosition.x) < 0.01 ? 1.0 : 0.0;
    

    
    float2 dz_dxy = {dfdx(in.worldSpacePosition.y), dfdy(in.worldSpacePosition.y)};
    float2 dx_dxy = {dfdx(in.worldSpacePosition.x), dfdy(in.worldSpacePosition.x)};
    float2 dxz = {length(dx_dxy), length(dz_dxy)};
//    float loda = mod(in.worldSpacePosition.y, gridSize) / abs(4 * length(dxz));
    float2 change = abs(fmod(in.worldSpacePosition.xy, {gridSize, gridSize})) / (4 * (dxz));
    float loda = max((1 - abs(saturate(change * 2 - 1))));
    float4 col = Thickcolour;
    col.a *= loda;
    if (col.a < 0.01) discard_fragment();
    
    if (xAxis > 0.0) {
        col = float4(xAxisColor, 1.0);
    }
//    if (zAxis > 0.0) {
//        col = float4(zAxisColor, 1.0);
//    }
    return col;
}


struct vertexOut {
    float4 position [[position]];
    float4  color;
};

struct pointCloudVertexOut {
    float4 position [[position]];
    float pointSize [[point_size]];
    float4  color;
};

struct MeshPointCloudVertexOut {
    float4 position [[position]];
    float3 normal;
    float4  color;
    float2 textCoord;
    float pointSize [[point_size]];
};

struct vertexOutLighting {
    float4 position [[position]];
    float3 WorldPos;
    float3 local_pos;
    float3 normal;
    float4 color;
    float2 textCoord;
};

struct Vertex3D {
    float3 position [[attribute(0)]];
    float4 colour [[attribute(1)]];
    float2 textCoord [[attribute(2)]];
    float3 normal [[attribute(3)]];
};

simd_float4 constant favCol[3] = {{0.8687, 0.4099, 0.9302, 1.0}, {0.4412, 0.2952, 0.9600, 1.0}, {0.21545,  0.74170,  0.64551,  1.00000}};

vertex vertexOut basicVertexShader(Vertex3D vertexElement [[stage_in]], constant float4x4& transform [[buffer(1)]],  constant float4x4& cam [[buffer(2)]]) {
    vertexOut vertOut = vertexOut();
    vertOut.color = vertexElement.colour;
    vertOut.position =  cam * transform * float4(vertexElement.position, 1);
//    vertOut.position /= vertOut.position.w;
//    vertOut.position.z -= 0.2;
    
    return vertOut;
}

vertex vertexOut instanceVertexShader(Vertex3D vertexElement [[stage_in]], constant float4x4& transform [[buffer(1)]],  constant float4x4& cam [[buffer(2)]], device const float4x4* instanceData [[buffer(3)]], uint instanceId [[instance_id]]) {
    vertexOut vertOut = vertexOut();
    vertOut.color = vertexElement.colour;
    vertOut.position = cam * instanceData[instanceId] * transform * float4(vertexElement.position, 1.0);
    return vertOut;
}

//vertex vertexOut NodeVertexShader(Vertex3D vertexElement [[stage_in]], constant float4x4& modalTransform [[buffer(1)]], constant float4x4& globalTransform [[buffer(2)]], constant float4x4& cam [[buffer(3)]], device const float4x4* instanceData [[buffer(4)]], uint instanceId [[instance_id]]) {
//    vertexOut vertOut = vertexOut();
//    vertOut.color = vertexElement.colour;
//    vertOut.position = cam * globalTransform * instanceData[instanceId] * modalTransform * float4(vertexElement.position, 1.0);
//    return vertOut;
//}

vertex vertexOutLighting NodeVertexShader(Vertex3D vertexElement [[stage_in]], constant float4x4& modalTransform [[buffer(1)]], constant float4x4& cam [[buffer(2)]], device const float4x4* instanceData [[buffer(3)]], uint instanceId [[instance_id]]) {
    vertexOutLighting vertOut = vertexOutLighting();
    vertOut.color = vertexElement.colour;
//    vertOut.position = cam * instanceData[instanceId] * modalTransform * float4(vertexElement.position, 1.0);
    vertOut.position = cam * instanceData[instanceId] * float4(vertexElement.position, 1.0);
    float3x3 camNorm = float3x3(
                            cam.columns[0].xyz, // First column without the 4th element
                            cam.columns[1].xyz, // Second column without the 4th element
                            cam.columns[2].xyz  // Third column without the 4th element
                                             );
    vertOut.normal =  camNorm * vertexElement.normal;
    vertOut.WorldPos = vertOut.position.xyz / vertOut.position.w;
    vertOut.textCoord = vertexElement.textCoord;
    float3 light = {0, 0, -1000};
//    vertOut.textCoord[0] = dot(vertexElement.normal, normalize(vertexElement.position - cam.columns[3].xyz));
    return vertOut;
}

vertex vertexOutLighting BillboardNodeVertexShader(Vertex3D vertexElement [[stage_in]], constant float4x4& modalTransform [[buffer(1)]], constant float4x4& cam [[buffer(2)]], device const float4x4* instanceData [[buffer(3)]], constant float3& cameraPos [[buffer(4)]], uint instanceId [[instance_id]]) {
    vertexOutLighting vertOut;
        vertOut.color = vertexElement.colour;
        vertOut.textCoord = vertexElement.textCoord;

        // 1. Get Instance Properties
        // We assume Column 3 is Position and Column 1 (Y-axis) is the Arrow/Spine Direction
        float4x4 instanceMat = instanceData[instanceId];
        float3 worldOrigin = instanceMat.columns[3].xyz;
        float3 spineDir = normalize(instanceMat.columns[1].xyz);
        
        // Scale extraction (optional, if your matrix has scaling)
        float lengthScale = length(instanceMat.columns[1].xyz);
        float widthScale  = length(instanceMat.columns[0].xyz);

        // 2. Calculate Billboarding Vectors
        float3 viewVec = normalize(cameraPos - worldOrigin);
        
        // Compute the 'Right' vector: Perpendicular to both Spine and View
        // This forces the arrow's width to always face the camera
        float3 billboardRight = cross(spineDir, viewVec);
        
        // Safety check: if looking directly down the spine, use a fallback
        if (length_squared(billboardRight) < 0.001) {
            billboardRight = cross(spineDir, float3(1, 0, 0));
        }
        billboardRight = normalize(billboardRight);
        
        // Compute the 'Normal' vector: Perpendicular to Spine and Right
        // This points out of the flat face of the arrow
        float3 billboardNormal = cross(billboardRight, spineDir);

        // 3. Construct the New World Position
        // We manually assemble the vertex position using our new basis vectors.
        // X controls width (along billboardRight)
        // Y controls length (along spineDir)
        float3 finalWorldPos = worldOrigin
                               + (spineDir * vertexElement.position.y * lengthScale)
                               + (billboardRight * vertexElement.position.x * widthScale);

        // 4. Output
        vertOut.position = cam * float4(finalWorldPos, 1.0);
        vertOut.WorldPos = finalWorldPos;
        
        // For the normal, we usually want it to point at the camera for a billboard
        // or use the computed billboardNormal.
        vertOut.normal = billboardNormal;

        return vertOut;
}


vertex pointCloudVertexOut pointCloudNodeVertexShader(Point3D pointElement [[stage_in]], constant float4x4& transform [[buffer(1)]],  constant float4x4& cam [[buffer(2)]], device const float4x4* instanceData [[buffer(3)]], uint instanceId [[instance_id]]) {
    pointCloudVertexOut vertOut = pointCloudVertexOut();
    vertOut.color = pointElement.colour;
    vertOut.pointSize = 7;
    vertOut.position = cam * instanceData[instanceId] * transform * float4(pointElement.position, 1.0);
    
    return vertOut;
}

vertex vertexOutLighting CustomShader1(Vertex3D vertexElement [[stage_in]], constant float4x4& modalTransform [[buffer(1)]], constant float4x4& cam [[buffer(2)]], device const float4x4* instanceData [[buffer(3)]], device const float4* colourbuff [[buffer(4)]], uint instanceId [[instance_id]]) {
    vertexOutLighting vertOut = vertexOutLighting();
    vertOut.color = colourbuff[instanceId];
//    vertOut.position = cam * instanceData[instanceId] * modalTransform * float4(vertexElement.position, 1.0);
    vertOut.position = cam * instanceData[instanceId] * float4(vertexElement.position, 1.0);
    float3x3 camNorm = float3x3(
                            cam.columns[0].xyz, // First column without the 4th element
                            cam.columns[1].xyz, // Second column without the 4th element
                            cam.columns[2].xyz  // Third column without the 4th element
                                             );
    vertOut.normal =  camNorm * vertexElement.normal;
    vertOut.WorldPos = vertOut.position.xyz / vertOut.position.w;
    vertOut.textCoord = vertexElement.textCoord;
    float3 light = {0, 0, -1000};
//    vertOut.textCoord[0] = dot(vertexElement.normal, normalize(vertexElement.position - cam.columns[3].xyz));
    return vertOut;
}


vertex pointCloudVertexOut pointCloudVertexShader(Point3D pointElement [[stage_in]], constant float4x4& transform [[buffer(1)]],  constant float4x4& cam [[buffer(2)]]) {
    pointCloudVertexOut vertOut = pointCloudVertexOut();
    vertOut.color = pointElement.colour;
    vertOut.pointSize = 10;
    vertOut.position = cam * transform * float4(pointElement.position, 1.0);
    
    return vertOut;
}



vertex MeshPointCloudVertexOut MeshPointCloudVertexShader(Vertex3D pointElement [[stage_in]], constant float4x4& transform [[buffer(1)]],  constant float4x4& cam [[buffer(2)]]) {
    MeshPointCloudVertexOut vertOut = MeshPointCloudVertexOut();
    vertOut.color = pointElement.colour;
    vertOut.pointSize = 10;
    vertOut.position = cam * transform * float4(pointElement.position, 1.0);
    vertOut.normal = pointElement.normal;
    vertOut.textCoord = pointElement.textCoord;
    
    return vertOut;
}

fragment float4 basicFragmentShader(vertexOut inColor  [[ stage_in ]])
{
    return inColor.color;
}

fragment float4 pointCloudFragmentShader(pointCloudVertexOut inColor  [[ stage_in ]])
{
    return inColor.color;
}

fragment float4 meshPointCloudFragmentShader(MeshPointCloudVertexOut inColor  [[ stage_in ]])
{
    return float4(inColor.normal, 1);
}


vertex vertexOutLighting lightingVertexShader(Vertex3D vertexElement [[stage_in]], constant float4x4& transform [[buffer(1)]],  constant float4x4& camera [[buffer(2)]]) {
    
    vertexOutLighting vertOut = vertexOutLighting();
    vertOut.color = vertexElement.colour;
    vertOut.position = camera * transform * float4(vertexElement.position, 1);
    float3x3 cam = float3x3(
                            camera.columns[0].xyz, // First column without the 4th element
                            camera.columns[1].xyz, // Second column without the 4th element
                            camera.columns[2].xyz  // Third column without the 4th element
                                             );
    vertOut.normal = ((cam * vertexElement.normal));
    vertOut.textCoord = vertexElement.textCoord;
    return vertOut;
}
float reLU(float x) {
    return x>0 ? x : 0;
}
fragment float4 lightingFragmentShader(vertexOutLighting inColor  [[ stage_in ]], constant bool& isTextured [[buffer(0)]], texture2d<float, access::sample> textureIn [[ texture(0) ]])
{
    float3 light = {0, 0, -1000};
    if (isTextured) {
        constexpr sampler colorSampler(address::clamp_to_edge, filter::linear);
        float4 sample = textureIn.sample(colorSampler, inColor.textCoord);
        return  sample;
    }
    return inColor.color;
//    return (inColor.color) * reLU(inColor.textCoord[0]) + 0.1;
    return (inColor.color) * reLU(dot(inColor.normal, normalize(inColor.position.xyz - light))) + 0.1;
    return (inColor.color+1.0) * dot(inColor.normal, normalize(inColor.position.xyz - light));
}



kernel void BlendCompute(device const uint8_t* A [[buffer(0)]], device const uint8_t* B [[buffer(1)]], device uint8_t* C [[buffer(2)]], constant size_t& size [[buffer(3)]], uint gid [[thread_position_in_grid]]) {
    
    C[4 * gid + 0] = (A[4 * gid + 0] * A[4 * gid + 3]  + B[4 * gid + 0] * (255 - A[4 * gid + 3])) / 255;
    C[4 * gid + 1] = (A[4 * gid + 1] * A[4 * gid + 3]  + B[4 * gid + 1] * (255 - A[4 * gid + 3])) / 255;
    C[4 * gid + 2] = (A[4 * gid + 2] * A[4 * gid + 3]  + B[4 * gid + 2] * (255 - A[4 * gid + 3])) / 255;
    
    C[4 * gid + 3] = B[4 * gid + 3];
}

kernel void InvertImgCompute(device uint8_t* A [[buffer(0)]], constant bool& evenAlpha [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    
    A[4 * gid + 0] = 255 - A[4 * gid + 0];
    A[4 * gid + 1] = 255 - A[4 * gid + 1];
    A[4 * gid + 2] = 255 - A[4 * gid + 2];
    if (evenAlpha) {
        A[4 * gid + 3] = 255 - A[4 * gid + 3];
    } else {
        A[4 * gid + 3] = A[4 * gid + 3];
    }
    
}

kernel void ChromaKeyCompute(device uint8_t* A [[buffer(0)]], constant simd_packed_char3& key [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (all(*((device simd_packed_char3*)(A + 4 * gid)) == key)) {
        A[4 * gid + 3] = 0;
    }
}

//kernel void AddGPU_F(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device float* C [[buffer(2)]],uint gid [[thread_position_in_grid]]) {
//    C[gid] = A[gid] + B[gid];
//}
//
//kernel void AddGPU_I(device const int* A [[buffer(0)]], device const int* B [[buffer(1)]], device int* C [[buffer(2)]],uint gid [[thread_position_in_grid]]) {
//    C[gid] = A[gid] + B[gid];
//}

kernel void AddGPU_C(device const uint8_t* A [[buffer(0)]], device const uint8_t* B [[buffer(1)]], device uint8_t* C [[buffer(2)]],uint gid [[thread_position_in_grid]]){
    C[4* gid + 0] = clamp((int)(A[4* gid + 0]) + (int)(B[4* gid + 0]), 0, 255);
    C[4* gid + 1] = clamp((int)(A[4* gid + 1]) + (int)(B[4* gid + 1]), 0, 255);
    C[4* gid + 2] = clamp((int)(A[4* gid + 2]) + (int)(B[4* gid + 2]), 0, 255);
    C[4* gid + 3] = 255;
}


kernel void SubGPU_F(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device float* C [[buffer(2)]],uint gid [[thread_position_in_grid]]) {
    C[gid] = A[gid] - B[gid];
}

kernel void SubGPU_I(device const int* A [[buffer(0)]], device const int* B [[buffer(1)]], device int* C [[buffer(2)]],uint gid [[thread_position_in_grid]]) {
    C[gid] = A[gid] - B[gid];
}

kernel void SubGPU_C(device const uint8_t* A [[buffer(0)]], device const uint8_t* B [[buffer(1)]], device uint8_t* C [[buffer(2)]], uint gid [[thread_position_in_grid]]){
    C[gid] = A[gid] - B[gid];
}


kernel void MulGPU_All(device const void* A [[buffer(0)]], device const void* B [[buffer(1)]], device void* C [[buffer(2)]], constant int& type [[buffer(3)]], constant size_t& stride [[buffer(4)]], constant size_t& strideI [[buffer(5)]] ,uint gid [[thread_position_in_grid]]){
    switch (type) {
        case 0: {
            auto a = (device int*)A;
            auto b = (device int*)B;
            auto c = (device int*)C;
            c[gid] = a[gid] * b[gid % stride];
            break;
        }
        case 1: { // float
            auto a = (device float*)A;
            auto b = (device float*)B;
            auto c = (device float*)C;
//            c[gid + strideI*(gid % stride)] = a[gid + strideI*(gid % stride)] * b[gid % stride];
            c[gid] = a[gid] * b[(gid / strideI) % stride];
            break;
        }
        case 2: { // image
            auto a = (device uint8_t*)A;
            auto b = (device uint8_t*)B;
            auto c = (device uint8_t*)C;
//            c[gid] = a[gid] * b[gid % stride + strideI];
            c[gid] = a[gid] * b[(gid / strideI) % stride];
            break;
        }
    }
}




kernel void ConversionGPU_All(device const void* A [[buffer(0)]], device const void*B [[buffer(1)]], constant int& type [[buffer(2)]], constant int& from [[buffer(3)]] ,uint gid [[thread_position_in_grid]]){
    switch (type) {
        case 0: { // image to float
            auto a = (device float*)A;
            auto b = (device uint8_t*)B;
            a[gid] = (float)(b[gid]);
            break;
        }
        case 1: { // float to image
            auto a = (device uint8_t*)A;
            auto b = (device float*)B;
            a[gid] = b[gid];
            break;
        }
        case 2: { // int to float
            auto a = (device float*)A;
            auto b = (device int*)B;
            a[gid] = b[gid];
        }
        case 3: { // int16 to int8
            auto a = (device uint8_t*)A;
            auto b = (device int16_t*)B;
            a[gid] = b[gid];
        }
    }
    
//    switch (from) {
//        case 0:  // from float
//            auto a = (device float*)A;
//            break;
//        case 1:  // image
//            auto a = (device uint8_t*)A;
//            break;
//        case 2:  // int
//            auto a = (device int*)A;
//            break;
//        case 3:  // int16 to int8
//            auto a = (device uint16_t*)A;
//            break;
//        case 4:  // simd_float2
//            auto a = (device simd_float2*)A;
//            break;
//        case 5:  // simd_float3
//            auto a = (device simd_float3*)A;
//            break;
//        case 6:  // simd_float3
//            auto a = (device simd_float4*)A;
//            break;
//    }
//    
//    switch (type) {
//        case 0:  // from float
//            auto b = (device float*)B;
//            b[gid] = (float)(a[gid]);
//            break;
//        case 1:  // image
//            auto b = (device uint8_t*)B;
//            b[gid] = (uint8_t)(a[gid]);
//            break;
//        case 2:  // int
//            auto b = (device int*)B;
//            b[gid] = (int)(a[gid]);
//            break;
//        case 3:  // int16
//            auto b = (device uint16_t*)B;
//            b[gid] = (int16)(a[gid]);
//            break;
//        case 4:  // int16 to int8
//            auto b = (device simd_float2*)B;
//            b[gid] = (simd_float2)(a[gid]);
//            break;
//        case 5:  // int16 to int8
//            auto b = (device simd_float3*)B;
//            b[gid] = (simd_float3)(a[gid]);
//            break;
//        case 6:  // int16 to int8
//            auto b = (device simd_float4*)B;
//            b[gid] = (simd_float4)(a[gid]);
//            break;
//    }
}



kernel void DerivativeGPU_All(device const void* A [[buffer(0)]], device const void* B [[buffer(1)]], constant size_t& stride [[buffer(2)]], constant size_t& max [[buffer(3)]],constant int& lastResolve [[buffer(4)]] , constant int& type [[buffer(5)]] ,uint gid [[thread_position_in_grid]]){
    switch (type) {
        case 0: { // float
            auto a = (device float*)A;
            auto b = (device float*)B;
            // gid + stride > max * ((gid / max) + 1)
            if ((gid / stride) < max) {
                a[gid] = b[gid + stride] - b[gid];
            } else {
                switch (lastResolve) {
                    case 0:
                        a[gid] = b[gid] - b[gid - stride];
                        break;
                    case 1:
                        a[gid] = b[gid - stride * max] - b[gid];
                        break;
                    case 2:
                        a[gid] = 0 - b[gid];
                        break;
                    case 3:
                        break;
                        
                }
            }
            break;
        }
        case 1: { // int
            auto a = (device int*)A;
            auto b = (device int*)B;
            if ((gid / stride) < max) {
                a[gid] = b[gid + stride] - b[gid];
            } else {
                switch (lastResolve) {
                    case 0:
                        a[gid] = b[gid] - b[gid - stride];
                        break;
                    case 1:
                        a[gid] = b[gid - stride * max] - b[gid];
                        break;
                    case 2:
                        a[gid] = 0 - b[gid];
                        break;
                    case 3:
                        break;
                }
            }
            break;
        }
        case 2: { // image
            auto a = (device uint8_t*)A;
            auto b = (device uint8_t*)B;
            if ((gid / stride) < max) {
                a[gid] = b[gid + stride] - b[gid];
            } else {
                switch (lastResolve) {
                    case 0:
                        a[gid] = b[gid] - b[gid - stride];
                        break;
                    case 1:
                        a[gid] = b[gid - stride * max] - b[gid];
                        break;
                    case 2:
                        a[gid] = 0 - b[gid];
                        break;
                    case 3:
                        break;
                }
            }
            break;
        }
        case 3: { // image
            auto a = (device float2*)A;
            auto b = (device float2*)B;
            if ((gid / stride) < max) {
                a[gid] = b[gid + stride] - b[gid];
            } else {
                switch (lastResolve) {
                    case 0:
                        a[gid] = b[gid] - b[gid - stride];
                        break;
                    case 1:
                        a[gid] = b[gid - stride * max] - b[gid];
                        break;
                    case 2:
                        a[gid] = 0 - b[gid];
                        break;
                    case 3:
                        break;
                }
            }
            break;
        }
    }
}






struct Uniforms {
    float4x4 modelViewProjectionMatrix;
    float4x4 inverseModelViewProjectionMatrix;
    float3 cameraPosition;
    float gridScale;
};

struct VertexOut {
    float4 position [[position]];
    float3 worldPosition;
    float3 viewDirection;
};

vertex VertexOut gridVertexShader(uint vertexID [[vertex_id]],
                                   constant float4x4& cam [[buffer(0)]], constant float3& worldCamPosition [[buffer(1)]], constant float4x4& inverseModelViewProjectionMatrix [[buffer(2)]]) {
    // Generate fullscreen quad vertices
    float2 positions[4] = {
        float2(-1.0, -1.0),
        float2( 1.0, -1.0),
        float2(-1.0,  1.0),
        float2( 1.0,  1.0)
    };
    
    float2 pos = positions[vertexID];
    
    VertexOut out;
    out.position = float4(pos, 0.7, 1.0);
    
    // Calculate world position by unprojecting screen coordinates
    float4x4 invMVP = inverseModelViewProjectionMatrix;
    float4 nearPoint = invMVP * float4(pos.x, pos.y, -1.0, 1.0);
    float4 farPoint = invMVP * float4(pos.x, pos.y, 1.0, 1.0);
    
    nearPoint /= nearPoint.w;
    farPoint /= farPoint.w;
    
    out.viewDirection = normalize(farPoint.xyz - nearPoint.xyz);
    out.worldPosition = nearPoint.xyz;
    
    return out;
}

fragment float4 gridFragmentShader(VertexOut in [[stage_in]],
                                constant float3& cameraPosition [[buffer(0)]], constant float4x4& inverseModelViewProjectionMatrix [[buffer(1)]]) {
    float gridScale = 0.1;
    // Ray-plane intersection to find ground position
    
    float3 rayDir = normalize(in.viewDirection);
    float3 rayOrigin = cameraPosition;
    
    // Intersect with XZ plane (y = 0)
    float t = -rayOrigin.y / rayDir.y;
    
    if (t < 0.0) {
        discard_fragment();
    }
    
    float3 worldPos = rayOrigin + rayDir * t;
    
    // Scale the grid
    float2 coord = worldPos.xz * gridScale;
    
    // Create grid lines using derivative-based anti-aliasing
    float2 grid = abs(fract(coord - 0.5) - 0.5) / fwidth(coord);
    float line = min(grid.x, grid.y);
    
    // Major grid lines every 10 units
    float2 majorGrid = abs(fract(coord * 0.1 - 0.5) - 0.5) / fwidth(coord * 0.1);
    float majorLine = min(majorGrid.x, majorGrid.y);
    
    // Create axis lines (X and Z axes in different colors)
    float xAxis = abs(worldPos.z) < 0.1 ? 1.0 : 0.0;
    float zAxis = abs(worldPos.x) < 0.1 ? 1.0 : 0.0;
    
    // Combine lines
    float gridIntensity = 1.0 - min(line, 1.0);
    float majorGridIntensity = 1.0 - min(majorLine, 1.0);
    
    // Distance-based fade
    float distance = length(worldPos - cameraPosition);
    float fade = exp(-distance * 0.02);
    float nearFade = smoothstep(0.0, 5.0, distance); // Fade out very close grid
    
    // Grid colors
    float3 gridColor = float3(0.3, 0.3, 0.4);
    float3 majorGridColor = float3(0.5, 0.5, 0.6);

    
    // Combine colors
    float3 finalColor = gridColor;
    float totalAlpha = gridIntensity;
    
    // Add major grid
    if (majorGridIntensity > 0.0) {
        finalColor = mix(finalColor, majorGridColor, majorGridIntensity * 0.7);
        totalAlpha = max(totalAlpha, majorGridIntensity * 0.8);
    }
    
    // Add axis lines
    if (xAxis > 0.0) {
        finalColor = mix(finalColor, xAxisColor, 0.8);
        totalAlpha = max(totalAlpha, 0.9);
    }
    if (zAxis > 0.0) {
        finalColor = mix(finalColor, zAxisColor, 0.8);
        totalAlpha = max(totalAlpha, 0.9);
    }
    
    // Apply fading
    float alpha = totalAlpha * fade * nearFade;
    
    // Ensure minimum visibility for main axes
    if (xAxis > 0.0 || zAxis > 0.0) {
        alpha = max(alpha, 0.3 * fade);
    }
    
    return float4(finalColor, alpha);
}

vertex vertexOutLighting DAGNodeVertexShader(uint vertexID [[vertex_id]],
                                     device const packed_float3* vert_position [[buffer(0)]],
                                     device const float4x4* instance_buffer [[buffer(1)]],
                                     constant float4x4& cam [[buffer(2)]],
                                     device const float4* color_buffer [[buffer(3)]],
                                     constant uint2& color_strides [[buffer(4)]],
                                     device const float2* uv_buffer [[buffer(5)]],
                                     device const packed_float3* normal_buffer [[buffer(6)]],
                                     constant uint& has_normals [[buffer(7)]],
                                     uint instanceId [[instance_id]]) 
{
    vertexOutLighting vertOut;
    
    vertOut.color = color_buffer[instanceId * color_strides.x + vertexID * color_strides.y];
    
    if (uv_buffer) {
        vertOut.textCoord = uv_buffer[vertexID];
    } else {
        vertOut.textCoord = float2(0.0, 0.0);
    }
    
    float3 pos = vert_position[vertexID];
    
    float4 world_pos = instance_buffer[instanceId] * float4(pos, 1.0);

    // Normals go through the cofactor matrix of the instance's 3x3 (the inverse-transpose times the
    // determinant), so non-uniform scale keeps them perpendicular to the surface. Zero = none: the
    // fragment shader then uses the flat face normal.
    if (has_normals) {
        float4x4 w = instance_buffer[instanceId];
        float3 a = w[0].xyz, b = w[1].xyz, c = w[2].xyz;
        float3x3 cofactor = float3x3(cross(b, c), cross(c, a), cross(a, b));
        vertOut.normal = cofactor * float3(normal_buffer[vertexID]);
    } else {
        vertOut.normal = float3(0.0);
    }
    vertOut.WorldPos = world_pos.xyz;
    vertOut.local_pos = pos;
    vertOut.position = cam * world_pos;
    
    return vertOut;
}

// Shadow pass: casters' positions as the light sees them. Depth only, no fragment shader.
vertex float4 shadow_vertex(uint vertexID [[vertex_id]], uint instanceId [[instance_id]],
                            device const packed_float3* vert_position [[buffer(0)]],
                            device const float4x4* instance_buffer [[buffer(1)]],
                            constant float4x4& light_vp [[buffer(2)]]) {
    return light_vp * (instance_buffer[instanceId] * float4(float3(vert_position[vertexID]), 1.0));
}

// How much of the shadow-casting light reaches world_pos: 1 lit, 0 in shadow, in between at the soft
// edges. Projects the point into the shadow map and compares depths: 3x3 samples, each a hardware 2x2
// filtered comparison (PCF).
static float shadow_visibility(depth2d<float, access::sample> map, float3 world_pos, float4x4 light_vp, float n_dot_l) {
    constexpr sampler cmp(coord::normalized, filter::linear, address::clamp_to_edge, compare_func::less_equal);
    float4 p = light_vp * float4(world_pos, 1.0);
    float3 ndc = p.xyz / p.w;
    if (any(abs(ndc.xy) > 1.0)) return 1.0;                        // beside the light's box: no caster there
    float2 uv = float2(ndc.x, -ndc.y) * 0.5 + 0.5;                 // NDC y is up, texture v is down
    float bias = mix(0.003, 0.0005, saturate(n_dot_l));             // surfaces at grazing angles need more
    // The box is fitted to the casters, so a receiver beyond its far end (the floor under a ball in the air)
    // is behind all of them: clamp to the far end, so it's shadowed wherever a caster was recorded and lit
    // where the map is still clear (1.0).
    float depth = min(ndc.z - bias, 1.0);
    float2 texel = 1.0 / float2(map.get_width(), map.get_height());
    float lit = 0.0;
    for (int y = -1; y <= 1; y++)
        for (int x = -1; x <= 1; x++)
            lit += map.sample_compare(cmp, uv + float2(x, y) * texel, depth);
    return lit / 9.0;
}

// ACES filmic tone mapping: Stephen Hill's fit of the ACES reference transform (RRT + ODT), the same
// curve and matrices three.js uses for ACESFilmicToneMapping. Linear in, linear out (the sRGB
// framebuffer encodes afterwards).
static float3 aces_rrt_odt_fit(float3 v) {
    float3 a = v * (v + 0.0245786) - 0.000090537;
    float3 b = v * (0.983729 * v + 0.4329510) + 0.238081;
    return a / b;
}
static float3 aces_filmic(float3 color) {
    const float3x3 aces_input = float3x3(float3(0.59719, 0.07600, 0.02840),     // sRGB -> ACES AP1, columns
                                         float3(0.35458, 0.90834, 0.13383),
                                         float3(0.04823, 0.01566, 0.83777));
    const float3x3 aces_output = float3x3(float3( 1.60475, -0.10208, -0.00327),  // AP1 -> sRGB, columns
                                          float3(-0.53108,  1.10813, -0.07276),
                                          float3(-0.07367, -0.00605,  1.07602));
    return saturate(aces_output * aces_rrt_odt_fit(aces_input * color));
}

// Neutral tone mapping: the highlight compression of Khronos PBR Neutral (three.js's NeutralToneMapping).
// Colours whose brightest channel is below 0.76 pass through unchanged; brighter ones are compressed
// toward 1 and slightly desaturated at the very top. Keeps base / brand colours accurate. Linear in,
// linear out. Unlike the Khronos spec, there's no black-level offset ("toe"): it subtracts the same
// amount from every channel of a dark colour, which on a dark scene crushes near-greys and pushes them
// toward their strongest channel (a dark cool floor turned visibly blue).
static float3 neutral_tone_map(float3 color) {
    const float start_compression = 0.8 - 0.04;
    const float desaturation = 0.15;
    float peak = max(color.r, max(color.g, color.b));
    if (peak < start_compression) return color;
    float d = 1.0 - start_compression;
    float new_peak = 1.0 - d * d / (peak + d - start_compression);
    color *= new_peak / peak;
    float g = 1.0 - 1.0 / (desaturation * (peak - new_peak) + 1.0);
    return mix(color, float3(new_peak), g);
}

// --- Physically based shading (Shading::Standard), as three.js's MeshStandardMaterial ---

// GGX / Trowbridge-Reitz normal distribution: how many microfacets face the half vector.
static float D_GGX(float n_dot_h, float alpha) {
    float a2 = alpha * alpha;
    float d = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (M_PI_F * d * d);
}

// Smith height-correlated visibility, G / (4 N.L N.V): microfacets shadowing / masking each other
// (three.js's V_GGX_SmithCorrelated).
static float V_SmithGGXCorrelated(float n_dot_l, float n_dot_v, float alpha) {
    float a2 = alpha * alpha;
    float gv = n_dot_l * sqrt(a2 + (1.0 - a2) * n_dot_v * n_dot_v);
    float gl = n_dot_v * sqrt(a2 + (1.0 - a2) * n_dot_l * n_dot_l);
    return 0.5 / max(gv + gl, 1e-6);
}

// Schlick Fresnel: reflectance rises toward 1 at grazing angles.
static float3 F_Schlick(float3 f0, float v_dot_h) {
    return f0 + (1.0 - f0) * pow(1.0 - v_dot_h, 5.0);
}

// Specular part of ambient light, until we have environment maps: Karis's analytic fit of the
// split-sum environment BRDF. Without it, metals under ambient light alone would be black.
static float3 env_brdf_approx(float3 f0, float roughness, float n_dot_v) {
    const float4 c0 = float4(-1.0, -0.0275, -0.572, 0.022);
    const float4 c1 = float4(1.0, 0.0425, 1.04, -0.04);
    float4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * n_dot_v)) * r.x + r.y;
    float2 ab = float2(-1.04, 1.04) * a004 + r.zw;
    return f0 * ab.x + ab.y;
}

// A Viewer light, as GPULight in SOPs/GeoNode.cpp packs it.
struct GPULight {
    float4 position_type;     // xyz position, w type (0 directional, 1 point)
    float4 direction_range;   // xyz direction the light travels (directional), w range (point: half-strength distance)
    float4 color_intensity;   // rgb colour, a intensity
};

fragment float4 DAGNodeFragmentShader(vertexOutLighting inColor [[stage_in]],
                                      constant bool& isTextured [[buffer(0)]],
                                      constant float3& clip_min [[buffer(1)]],
                                      constant float3& clip_max [[buffer(2)]],
                                      constant uint& shading [[buffer(3)]],        // Shading: 0 unlit, 1 lit, 2 standard
                                      constant float3& camera_pos [[buffer(4)]],
                                      constant GPULight* lights [[buffer(5)]],
                                      constant uint& light_count [[buffer(6)]],
                                      constant float3& ambient [[buffer(7)]],
                                      constant float2& surface [[buffer(8)]],      // specular strength, shininess
                                      constant float4x4& light_vp [[buffer(9)]],
                                      constant int& shadow_light [[buffer(10)]],   // index of the shadowed light, -1 none
                                      constant float2& tone [[buffer(11)]],         // tone mapping (0 none, 1 ACES, 2 neutral), exposure
                                      constant float2& pbr [[buffer(12)]],          // metallic, roughness (Standard)
                                      constant float4& fog_params [[buffer(13)]],   // mode (0 none, 1 linear, 2 exp2), near, far, density
                                      constant float3& fog_color [[buffer(14)]],
                                      constant uint& texture_repeat [[buffer(15)]],  // 1: UVs outside [0, 1] tile the texture
                                      texture2d<float, access::sample> textureIn [[texture(0)]],
                                      depth2d<float, access::sample> shadow_map [[texture(1)]])
{
    // Flat face normal from screen-space derivatives, for meshes without normals. Computed before any
    // branch or discard: derivatives need every pixel of the 2x2 quad.
    float3 face_normal = cross(dfdx(inColor.WorldPos), dfdy(inColor.WorldPos));

    if (any(inColor.WorldPos < clip_min) || any(inColor.WorldPos > clip_max)) {
        discard_fragment();
    }

    float4 color = inColor.color;
    if (isTextured) {
        constexpr sampler colorSampler(address::clamp_to_edge, filter::linear);
        constexpr sampler repeatSampler(address::repeat, filter::linear, mip_filter::linear);
        color *= texture_repeat ? textureIn.sample(repeatSampler, inColor.textCoord)
                                : textureIn.sample(colorSampler, inColor.textCoord);
    }
    // Fully transparent: drop it, so it doesn't write depth and hide whatever is drawn after it.
    if (color.a <= 0.0) {
        discard_fragment();
    }

    if (shading == 1 || shading == 2) {   // Lit (Blinn-Phong) or Standard (PBR)
        float3 N = length_squared(inColor.normal) > 1e-12 ? inColor.normal : face_normal;
        N = normalize(N);
        float3 V = normalize(camera_pos - inColor.WorldPos);
        if (dot(N, V) < 0.0) N = -N;   // two-sided: light whichever side faces the camera

        // Standard: split the colour into a diffuse part and a specular reflectance at normal incidence (F0).
        // Dielectrics reflect ~4% white; metals reflect their colour and have no diffuse.
        float metallic = saturate(pbr.x);
        float roughness = clamp(pbr.y, 0.0525, 1.0);          // three.js's floor: a perfect mirror aliases
        float alpha = roughness * roughness;                    // perceptual -> GGX roughness
        float3 diffuse_color = color.rgb * (1.0 - metallic);
        float3 f0 = mix(float3(0.04), color.rgb, metallic);
        float n_dot_v = max(dot(N, V), 1e-4);

        float3 lit = shading == 2 ? ambient * (diffuse_color + env_brdf_approx(f0, roughness, n_dot_v))
                                  : color.rgb * ambient;
        float3 phong_specular = float3(0.0);
        for (uint i = 0; i < light_count; i++) {
            GPULight light = lights[i];
            float3 radiance = light.color_intensity.rgb * light.color_intensity.a;
            float3 L;
            if (light.position_type.w < 0.5) {            // directional
                L = -light.direction_range.xyz;
            } else {                                      // point: falls to half strength at `range`
                float3 to_light = light.position_type.xyz - inColor.WorldPos;
                float d = length(to_light);
                L = to_light / max(d, 1e-5);
                float r = max(light.direction_range.w, 1e-5);
                radiance /= 1.0 + (d / r) * (d / r);
            }
            float n_dot_l = max(dot(N, L), 0.0);
            if (n_dot_l <= 0.0) continue;
            float visibility = (int(i) == shadow_light) ? shadow_visibility(shadow_map, inColor.WorldPos, light_vp, n_dot_l) : 1.0;
            float3 H = normalize(L + V);

            if (shading == 1) {        // Lit: Lambert + Blinn-Phong
                lit += color.rgb * radiance * n_dot_l * visibility;
                phong_specular += radiance * visibility * surface.x * pow(max(dot(N, H), 0.0), surface.y);
            } else {                   // Standard: Lambert + Cook-Torrance GGX
                float3 F = F_Schlick(f0, saturate(dot(V, H)));
                float3 specular = D_GGX(saturate(dot(N, H)), alpha) * V_SmithGGXCorrelated(n_dot_l, n_dot_v, alpha) * F;
                // Our lights' intensity means "a white Lambert surface facing it shows that brightness", i.e.
                // three.js's intensity / pi: so diffuse has no 1/pi, and the specular term carries a pi instead.
                lit += radiance * n_dot_l * visibility * (diffuse_color + M_PI_F * specular);
            }
        }
        color.rgb = lit + phong_specular;

        color.rgb *= tone.y;                                   // exposure
        if (uint(tone.x) == 1) {
            color.rgb = aces_filmic(color.rgb / 0.6);          // three.js scales by 1/0.6 so exposure 1 matches its look
        } else if (uint(tone.x) == 2) {
            color.rgb = neutral_tone_map(color.rgb);
        }
    }

    // Fog, last (after tone mapping, as three.js does): blend toward the fog colour with camera distance.
    uint fog_mode = uint(fog_params.x);
    if (fog_mode != 0) {
        float dist = length(camera_pos - inColor.WorldPos);
        float amount = fog_mode == 1 ? smoothstep(fog_params.y, fog_params.z, dist)
                                     : 1.0 - exp(-(fog_params.w * dist) * (fog_params.w * dist));
        color.rgb = mix(color.rgb, fog_color, saturate(amount));
    }
    return color;
}


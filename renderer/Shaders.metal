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
    
    float2 dz_dxy = {dfdx(in.worldSpacePosition.y), dfdy(in.worldSpacePosition.y)};
    float2 dx_dxy = {dfdx(in.worldSpacePosition.x), dfdy(in.worldSpacePosition.x)};
    float2 dxz = {length(dx_dxy), length(dz_dxy)};
//    float loda = mod(in.worldSpacePosition.y, gridSize) / abs(4 * length(dxz));
    float2 change = abs(fmod(in.worldSpacePosition.xy, {gridSize, gridSize})) / (4 * (dxz));
    float loda = max((1 - abs(saturate(change * 2 - 1))));
    float4 col = Thickcolour;
    col.a *= loda;
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

fragment float4 lightingFragmentShader(vertexOutLighting inColor  [[ stage_in ]], constant bool& isTextured [[buffer(0)]], texture2d<float, access::sample> textureIn [[ texture(0) ]])
{
    float3 light = {0, 0, -1000};
    if (isTextured) {
        constexpr sampler colorSampler(address::clamp_to_edge, filter::linear);
        float4 sample = textureIn.sample(colorSampler, inColor.textCoord);
        return  sample;
    }
    return (inColor.color);
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

kernel void AddGPU_F(device const float* A [[buffer(0)]], device const float* B [[buffer(1)]], device float* C [[buffer(2)]],uint gid [[thread_position_in_grid]]) {
    C[gid] = A[gid] + B[gid];
}

kernel void AddGPU_I(device const int* A [[buffer(0)]], device const int* B [[buffer(1)]], device int* C [[buffer(2)]],uint gid [[thread_position_in_grid]]) {
    C[gid] = A[gid] + B[gid];
}

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


kernel void ConversionGPU_All(device const void* A [[buffer(0)]], device const void* B [[buffer(1)]], constant int& type [[buffer(2)]], constant int& from [[buffer(2)]] ,uint gid [[thread_position_in_grid]]){
//    switch (type) {
//        case 0: { // image to float
//            auto a = (device float*)A;
//            auto b = (device uint8_t*)B;
//            a[gid] = (float)(b[gid]);
//            break;
//        }
//        case 1: { // float to image
//            auto a = (device uint8_t*)A;
//            auto b = (device float*)B;
//            a[gid] = b[gid];
//            break;
//        }
//        case 2: { // int to float
//            auto a = (device float*)A;
//            auto b = (device int*)B;
//            a[gid] = b[gid];
//        }
//        case 3: { // int16 to int8
//            auto a = (device uint8_t*)A;
//            auto b = (device int16_t*)B;
//            a[gid] = b[gid];
//        }
//    }
    
    switch (from) {
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
    float3 xAxisColor = float3(1.0, 0.3, 0.3); // Red for X axis
    float3 zAxisColor = float3(0.3, 0.3, 1.0); // Blue for Z axis
    
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

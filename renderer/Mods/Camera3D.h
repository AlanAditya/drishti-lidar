//
//  Camera3D.h
//  WorldOf3D
//
//  Created by Aditya Dudeja on 06/02/26.
//

#ifndef Camera3D_h
#define Camera3D_h

#include <simd/simd.h>
#include <print>

enum class TransformationMode {
    Orbit,
    Translate,
    Zoom
};

class Camera3D {
    simd_float3 oldPosition = {0.0, 0.0, -3.0};
    simd_float3 up = {0, 1, 0};
    simd_float3 right = {1, 0, 0};
    simd_float3 forward = {0, 0, 1};
    
    float azimuthalAngle = 0;
    float polarAngle = 0;
    
    // Projection parameters
    float fov = M_PI * 0.25f; // 45 degrees field of view
    float orthoSize = 10.0f;  // Size of orthographic view
    float aspectRatio = 1;
    
    // Camera Limits
    float farP = 1000;
    float nearP = 0.1;
    float minPolarAngle = -M_PI / 2 + 0.1;
    float maxPolarAngle = M_PI / 2 - 0.1;
    
    bool AxisUpdated = false;
    float scale = 1;
    float sensitivity = 1;

public:
    bool updated = false;
    bool isPerspective = true; // true = perspective, false = orthographic
    simd_float3 position = {0.0, 0.0, -3.00};
    simd_float3 target = {0, 0, 0};
    simd_float4x4 viewMatrix;
    simd_float4x4 inverseProjectionMatrix;
    
    Camera3D() {
        simd_float3 vec = target - position;
        forward = simd::normalize(vec);
        right = simd::normalize(simd::cross(forward, up));
        up = simd::normalize(simd::cross(right, forward));
        AxisUpdated = true;
        updateAngles(position);
    }
    
    void updateAngles(simd_float3 cameraPosition) {
        // camera is at {0, 0, -1} all angles are zero
        // polar angle is the angle with the negative z axis
        // azimuthal angle of r vec with xz plane
        
        float r = simd_length(cameraPosition);
        azimuthalAngle = asin(cameraPosition.y / r);
        polarAngle = atan2(cameraPosition.x, -cameraPosition.z); // becuase the camera is one the negative z axis we want to take the angle from -ve z axis so we add a -cameraPosition.z as by default atan2(x,z) gives angle from +ve z axis but the cam was on -ve so it was giving pi
        
    }
    
    void updateOLD() {
        oldPosition = position;
        scale = 1;
    }
    
    void updateMatrix() {
        if (!updated) {
            // Create view matrix (look-at)
            simd_float3 zAxis = simd::normalize(target - position); // Camera forward (towards viewer in RH)
            simd_float3 xAxis = simd::normalize(simd::cross(up, zAxis)); // Right
            simd_float3 yAxis = simd::normalize(simd::cross(zAxis, xAxis)); // Up
            
            // View matrix (right-handed)
            simd_float4 row0 = {xAxis.x, xAxis.y, xAxis.z, -simd_dot(xAxis, position)};
            simd_float4 row1 = {yAxis.x, yAxis.y, yAxis.z, -simd_dot(yAxis, position)};
            simd_float4 row2 = {zAxis.x, zAxis.y, zAxis.z, -simd_dot(zAxis, position)};
            simd_float4 row3 = {0,      0,      0,      1         };
            
            simd_float4x4 viewMat = simd_matrix_from_rows(row0, row1, row2, row3);
            
            // Create projection matrix
            simd_float4x4 projMat;
            if (isPerspective) {
                // Perspective projection matrix (right-handed, 0 to 1 depth)
                
                float yScale = 1.0f / tanf(fov * 0.5f);     // cotangent of half FOV
                float xScale = yScale / aspectRatio;
                float zRange = (farP - nearP);
                float zScale = farP / zRange;
                float wzScale = -farP * nearP / zRange;

                // Construct rows as SIMD4 vectors
                simd_float4 row0 = { xScale, 0,       0,      0 };
                simd_float4 row1 = { 0,       yScale, 0,      0 };
                simd_float4 row2 = { 0,       0,       zScale, wzScale };
                simd_float4 row3 = { 0,       0,       1,      0 };

                // Build the matrix from rows
                projMat = simd_matrix_from_rows(row0, row1, row2, row3);
            } else {
                // Orthographic projection matrix (0 to 1 depth)
                float right = orthoSize * aspectRatio * 0.5f;
                float left = -right;
                float top = orthoSize * 0.5f;
                float bottom = -top;
                float range = farP - nearP;
                
                simd_float4 proj0 = {2.0f / (right - left), 0, 0, 0};
                simd_float4 proj1 = {0, 2.0f / (top - bottom), 0, 0};
                simd_float4 proj2 = {0, 0, 1.0f / range, 0};
                simd_float4 proj3 = {0, 0, -nearP / range, 1};
                
                projMat = simd_matrix(proj0, proj1, proj2, proj3);
            }
            
//            std::cout << simd_transpose(viewMat) << "\n" << simd_transpose(projMat) << "\n";
            
            // Combine projection and view matrices
            viewMatrix = simd_mul(projMat, viewMat);
            inverseProjectionMatrix = inverseProjectionMat();
            updated = true;
        }
    }
    
    void handleMouseEvents(float deltaX, float deltaY, bool isRightMouseButton, bool isShiftPressed, TransformationMode eventType) {
            float sensitivity = 0.01f;
            
//        std::cout << polarAngle << " " << azimuthalAngle << "\n";
            if (isRightMouseButton || eventType == TransformationMode::Translate) {
                // Pan camera
                if (!AxisUpdated) {
                    simd_float3 vec = target - position;
                    
                    forward = simd::normalize(vec);
                    right = simd::normalize(simd::cross(forward, up));
                    up = simd::normalize(simd::cross(right, forward));
                    AxisUpdated = true;
                }
                auto dist = simd_length(target-position);
                auto dP = (up * deltaY + right * deltaX) * 0.05 * sensitivity * dist;
                target += dP;
                position += dP;
                updateOLD();
            } else if (eventType == TransformationMode::Orbit){
                // Orbit camera
                azimuthalAngle += deltaY * sensitivity;
//                polarAngle     += deltaX * sensitivity;

                // Update up vector based on azimuthal angle
                
                if (std::cos(azimuthalAngle) < 0.0f) {
                    up = simd::float3{0, -1, 0};
                    polarAngle += deltaX * sensitivity;
                } else {
                    up = simd::float3{0, 1, 0};
                    polarAngle -= deltaX * sensitivity;
                }
                // Calculate camera distance
                float distance = simd::length(target - position);

                // Optionally clamp polar angle
                // polarAngle = std::fmax(0.01f, std::fmin(static_cast<float>(M_PI) - 0.01f, polarAngle));
                // Calculate new camera position
                float x = target.x + distance * std::sin(polarAngle) * std::cos(azimuthalAngle) ;
                float y = target.y + distance * std::sin(azimuthalAngle) ;
                float z = target.z + distance * -std::cos(polarAngle) * std::cos(azimuthalAngle);

                // Update camera position
                position = simd::float3{x, y, z};
                updateOLD(); // Assuming it's defined
                AxisUpdated = false;
            } else if (eventType == TransformationMode::Zoom) {
                position = target + (oldPosition - target) * (1 / deltaX );
            }
            updated = false;
//            std::cout << position;
            
        }
    
    void setMouseEvent(float deltaX, float deltaY, bool isRightMouseButton, bool isShiftPressed, TransformationMode eventType) {
            float sensitivity = 0.01f;
            
//        std::cout << polarAngle << " " << azimuthalAngle << "\n";
            if (isRightMouseButton || eventType == TransformationMode::Translate) {
                // Pan camera
                if (!AxisUpdated) {
                    simd_float3 vec = target - position;
                    
                    forward = simd::normalize(vec);
                    right = simd::normalize(simd::cross(forward, up));
                    up = simd::normalize(simd::cross(right, forward));
                    AxisUpdated = true;
                }
                auto dist = simd_length(target-position);
                auto dP = (up * deltaY + right * deltaX) * 0.05 * sensitivity * dist;
                target = oldPosition +  dP;
                position = oldPosition + dP;
//                updateOLD();
            } else if (eventType == TransformationMode::Orbit){
                // Orbit camera
                azimuthalAngle = deltaY * sensitivity;
                polarAngle     = deltaX * sensitivity;

                // Update up vector based on azimuthal angle
                if (std::cos(azimuthalAngle) < 0.0f) {
                    up = simd::float3{0, -1, 0};
                } else {
                    up = simd::float3{0, 1, 0};
                }

                // Calculate camera distance
                float distance = simd::length(target - position);

                // Optionally clamp polar angle
                // polarAngle = std::fmax(0.01f, std::fmin(static_cast<float>(M_PI) - 0.01f, polarAngle));
                // Calculate new camera position
                float x = target.x + distance * std::sin(polarAngle) * std::cos(azimuthalAngle) ;
                float y = target.y + distance * std::sin(azimuthalAngle) ;
                float z = target.z + distance * -std::cos(polarAngle) * std::cos(azimuthalAngle);

                // Update camera position
                position = simd::float3{x, y, z};
                updateOLD(); // Assuming it's defined
                AxisUpdated = false;
            } else if (eventType == TransformationMode::Zoom) {
                position = target + (oldPosition - target) * (1 / deltaX );
            }
            updated = false;
//            std::cout << position;
            
        }
    
    void updateRawCamPosition(simd_float3 dPosition) {
        position += dPosition;
        target += dPosition;
        updated = false;
        
    }
    
    void updateAspectRatio(float ratio) {
        aspectRatio = ratio;
        updated = false;
    }
    
    void setFieldOfView(float fovRadians) {
        fov = fovRadians;
        updated = false;
    }
    
    void setOrthographicSize(float size) {
        orthoSize = size;
        updated = false;
    }
    
    void toggleProjection() {
        isPerspective = !isPerspective;
        updated = false;
        AxisUpdated = false;
    }
    
    simd_float4x4 inverseProjectionMat() {
        return  simd_inverse(viewMatrix);
    }
};




#endif /* Camera3D_h */

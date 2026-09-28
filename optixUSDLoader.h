#pragma once

#include <cstdint>
#include <optix.h>
#include <cuda_runtime.h>

// --- Host & Device Vector Math Operators for float3 ---
inline __host__ __device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

// Component-wise multiplication for two float3 vectors
inline __host__ __device__ float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

inline __host__ __device__ float3 operator*(const float3& a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

inline __host__ __device__ float3 operator*(float b, const float3& a) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

inline __host__ __device__ float3 operator/(const float3& a, float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}
// -----------------------------------------------------

struct Params
{
    uchar4*                 image;
    unsigned int            image_width;
    unsigned int            image_height;
    float3                  cam_eye;
    float3                  cam_u, cam_v, cam_w;
    OptixTraversableHandle  handle;

    // Geometry Buffers
    float3*                 vertices;
    float3*                 colors;
    float3*                 normals;
    float*                  opacities;
    uint32_t*               indices;

    // Lighting
    float3                  light_pos;
};

struct RayGenData {};
struct MissData {};
struct HitGroupData {};

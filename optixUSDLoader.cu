
#include <optix.h>
#include <cuda_runtime.h>
#include <cstdint>
#include "optixUSDLoader.h"

extern "C" {
__constant__ Params params;
}

// --- Inline Vector Math Operators (Device) ---
inline __device__ float dot(const float3& a, const float3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline __device__ float3 normalize(const float3& v) { float invLen = 1.0f / sqrtf(dot(v, v)); return v * invLen; }

inline __device__ bool refract(const float3& v, const float3& n, float ni_over_nt, float3& refracted) {
    float3 uv = normalize(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt * (1.0f - dt * dt);
    if (discriminant > 0.0f) {
        refracted = ni_over_nt * (uv - n * dt) - n * sqrtf(discriminant);
        return true;
    }
    return false;
}

inline __device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0) * powf((1.0f - cosine), 5.0f);
}
// ---------------------------------------------

extern "C" __global__ void __raygen__rg()
{
    const uint3    idx = optixGetLaunchIndex();
    const uint3    dim = optixGetLaunchDimensions();

    const float2 d = make_float2(
        (static_cast<float>(idx.x) + 0.5f) / static_cast<float>(dim.x),
        (static_cast<float>(idx.y) + 0.5f) / static_cast<float>(dim.y)
    );

    float3 ray_origin    = params.cam_eye;
    float3 ray_direction = normalize(params.cam_w + (d.x - 0.5f) * params.cam_u + (d.y - 0.5f) * params.cam_v);

    unsigned int p0 = 0, p1 = 0, p2 = 0, p3 = 0;

    optixTrace(
        params.handle, ray_origin, ray_direction, 0.0f, 1e16f, 0.0f,
        OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1, 0, p0, p1, p2, p3
    );

    float3 result_color = make_float3(__uint_as_float(p0), __uint_as_float(p1), __uint_as_float(p2));

    const uint32_t image_index = idx.y * dim.x + idx.x;
    params.image[image_index] = make_uchar4(
        static_cast<unsigned char>(fminf(result_color.x * 255.0f, 255.0f)),
        static_cast<unsigned char>(fminf(result_color.y * 255.0f, 255.0f)),
        static_cast<unsigned char>(fminf(result_color.z * 255.0f, 255.0f)),
        255
    );
}

extern "C" __global__ void __miss__ms()
{
    unsigned int flags = optixGetRayFlags();
    if (flags & OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT) {
        optixSetPayload_0(0);
    } else {
        optixSetPayload_0(__float_as_uint(0.08f));
        optixSetPayload_1(__float_as_uint(0.08f));
        optixSetPayload_2(__float_as_uint(0.12f));
    }
}

extern "C" __global__ void __closesthit__ch()
{
    unsigned int flags = optixGetRayFlags();
    if (flags & OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT) {
        optixSetPayload_0(1);
        return;
    }

    unsigned int depth = optixGetPayload_3();
    const unsigned int max_depth = 4;

    if (depth >= max_depth) {
        optixSetPayload_0(__float_as_uint(0.0f));
        optixSetPayload_1(__float_as_uint(0.0f));
        optixSetPayload_2(__float_as_uint(0.0f));
        return;
    }

    const uint32_t primIdx = optixGetPrimitiveIndex();
    const float2   bary    = optixGetTriangleBarycentrics();

    float3 baseColor = make_float3(0.8f, 0.8f, 0.8f);
    float  opacity   = 1.0f;

    uint32_t idx0 = params.indices[3 * primIdx + 0];
    uint32_t idx1 = params.indices[3 * primIdx + 1];
    uint32_t idx2 = params.indices[3 * primIdx + 2];

    float b0 = 1.0f - bary.x - bary.y;
    float b1 = bary.x;
    float b2 = bary.y;

    if (params.colors) {
        baseColor = params.colors[idx0] * b0 + params.colors[idx1] * b1 + params.colors[idx2] * b2;
    }

    if (params.opacities) {
        opacity = params.opacities[idx0] * b0 + params.opacities[idx1] * b1 + params.opacities[idx2] * b2;
    }

    // Uncomment the line below temporarily if your USD file defaults to opaque (1.0)
    // and you want to force test glass transparency right now:
    // opacity = 0.3f;

    float3 normal = make_float3(0.0f, 1.0f, 0.0f);
    if (params.normals) {
        normal = normalize(
            params.normals[idx0] * b0 +
            params.normals[idx1] * b1 +
            params.normals[idx2] * b2
        );
    }

    float3 rayOrig = optixGetWorldRayOrigin();
    float3 rayDir  = optixGetWorldRayDirection();
    float  tHit    = optixGetRayTmax();
    float3 hitPos  = rayOrig + rayDir * tHit;

    float3 finalColor = make_float3(0.0f, 0.0f, 0.0f);

    if (opacity < 0.99f) {
        bool frontFace = dot(rayDir, normal) < 0.0f;
        float3 outwardNormal = frontFace ? normal : normal * -1.0f;
        float ni_over_nt = frontFace ? (1.0f / 1.5f) : 1.5f;
        float cosine = frontFace ? -dot(rayDir, normal) : dot(rayDir, normal) * 1.5f;

        float reflectProb = schlick(cosine, 1.5f);

        float3 reflectedDir = rayDir - 2.0f * dot(rayDir, outwardNormal) * outwardNormal;
        float3 reflectOrig = hitPos + outwardNormal * 1e-3f;

        unsigned int rp0 = 0, rp1 = 0, rp2 = 0;
        unsigned int nextDepth = depth + 1;
        optixTrace(
            params.handle, reflectOrig, reflectedDir, 0.0f, 1e16f, 0.0f,
            OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1, 0, rp0, rp1, rp2, nextDepth
        );
        float3 reflectionColor = make_float3(__uint_as_float(rp0), __uint_as_float(rp1), __uint_as_float(rp2));

        float3 refractedDir;
        float3 transmissionColor = make_float3(0.0f, 0.0f, 0.0f);
        if (refract(rayDir, outwardNormal, ni_over_nt, refractedDir)) {
            float3 refractOrig = hitPos + outwardNormal * -1e-3f;
            unsigned int tp0 = 0, tp1 = 0, tp2 = 0;
            optixTrace(
                params.handle, refractOrig, refractedDir, 0.0f, 1e16f, 0.0f,
                OptixVisibilityMask(255), OPTIX_RAY_FLAG_NONE, 0, 1, 0, tp0, tp1, tp2, nextDepth
            );
            transmissionColor = make_float3(__uint_as_float(tp0), __uint_as_float(tp1), __uint_as_float(tp2));
        }

        finalColor = (reflectionColor * reflectProb + transmissionColor * (1.0f - reflectProb)) * baseColor;
    }
    else {
        float3 shadedPos = hitPos + normal * 1e-3f;

        float3 lightDirVec = params.light_pos - hitPos;
        float  lightDist = sqrtf(dot(lightDirVec, lightDirVec));
        float3 lightDir = lightDirVec / lightDist;

        unsigned int occluded = 0;
        optixTrace(
            params.handle, shadedPos, lightDir, 0.0f, lightDist - 1e-2f, 0.0f,
            OptixVisibilityMask(255), OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            0, 1, 0, occluded
        );

        float attenuation = 1.0f / (1.0f + 0.05f * lightDist + 0.01f * lightDist * lightDist);
        float3 ambient = baseColor * 0.15f;
        float3 diffuse = make_float3(0.0f, 0.0f, 0.0f);
        float3 specular = make_float3(0.0f, 0.0f, 0.0f);

        if (occluded == 0) {
            float diff = fmaxf(dot(normal, lightDir), 0.0f);
            diffuse = baseColor * diff * attenuation * 1.5f;

            float3 viewDir = normalize(params.cam_eye - hitPos);
            float3 halfwayDir = normalize(lightDir + viewDir);
            float spec = powf(fmaxf(dot(normal, halfwayDir), 0.0f), 32.0f);
            specular = make_float3(1.0f, 1.0f, 1.0f) * spec * attenuation * 0.8f;
        }

        finalColor = ambient + diffuse + specular;
    }

    optixSetPayload_0(__float_as_uint(finalColor.x));
    optixSetPayload_1(__float_as_uint(finalColor.y));
    optixSetPayload_2(__float_as_uint(finalColor.z));

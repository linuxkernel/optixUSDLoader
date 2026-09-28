#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>

#include <cuda_runtime.h>
#include <GLFW/glfw3.h>

#include <fstream>
#include <iostream>
#include <vector>
#include <array>
#include <cmath>

// OpenUSD Headers
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/primvar.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/shader.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec3d.h>

#include "optixUSDLoader.h"

template <typename T>
struct SbtRecord {
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>      RayGenRecord;
typedef SbtRecord<MissData>        MissRecord;
typedef SbtRecord<HitGroupData>    HitGroupRecord;

#define CUDA_CHECK( call ) \
    do { \
        cudaError_t error = call; \
        if( error != cudaSuccess ) { \
            fprintf( stderr, "CUDA error line %d: %s (%s)\n", __LINE__, cudaGetErrorString( error ), #call ); \
            exit( 2 ); \
        } \
    } while( 0 )

#define OPTIX_CHECK( call ) \
    do { \
        OptixResult res = call; \
        if( res != OPTIX_SUCCESS ) { \
            fprintf( stderr, "OptiX error line %d: %s (%s)\n", __LINE__, optixGetErrorName( res ), #call ); \
            exit( 2 ); \
        } \
    } while( 0 )

std::string readPTXFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Cannot open PTX file: " << filename << "\n";
        exit(1);
    }
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

bool loadUsdMesh(const std::string& filename, std::vector<float3>& vertices, std::vector<float3>& normals, std::vector<float3>& colors, std::vector<float>& opacities, std::vector<uint32_t>& indices) {
    pxr::UsdStageRefPtr stage = pxr::UsdStage::Open(filename);
    if (!stage) {
        std::cerr << "Failed to open USD file: " << filename << "\n";
        return false;
    }

    pxr::UsdGeomXformCache xformCache;
    bool foundAny = false;

    for (pxr::UsdPrim prim : stage->Traverse()) {
        if (prim.IsA<pxr::UsdGeomMesh>()) {
            pxr::UsdGeomMesh usdMesh(prim);

            pxr::VtArray<pxr::GfVec3f> points;
            usdMesh.GetPointsAttr().Get(&points);
            if (points.empty()) continue;

            pxr::VtArray<int> faceVertexCounts;
            usdMesh.GetFaceVertexCountsAttr().Get(&faceVertexCounts);

            pxr::VtArray<int> faceVertexIndices;
            usdMesh.GetFaceVertexIndicesAttr().Get(&faceVertexIndices);
            if (faceVertexCounts.empty() || faceVertexIndices.empty()) continue;

            pxr::GfMatrix4d transform = xformCache.GetLocalToWorldTransform(prim);

            pxr::GfVec3f defaultColor(0.8f, 0.8f, 0.8f);
            float defaultOpacity = 1.0f;

            pxr::VtArray<pxr::GfVec3f> displayColors;
            pxr::UsdGeomPrimvar colorPrimvar = usdMesh.GetDisplayColorPrimvar();
            bool hasDisplayColor = colorPrimvar.Get(&displayColors) && !displayColors.empty();

            pxr::VtArray<float> displayOpacities;
            pxr::UsdGeomPrimvar opacityPrimvar = usdMesh.GetDisplayOpacityPrimvar();
            bool hasDisplayOpacity = opacityPrimvar.Get(&displayOpacities) && !displayOpacities.empty();

            if (!hasDisplayColor) {
                pxr::UsdShadeMaterialBindingAPI bindingAPI(prim);
                pxr::UsdShadeMaterial material = bindingAPI.ComputeBoundMaterial();
                if (material) {
                    pxr::UsdShadeShader shader = material.ComputeSurfaceSource();
                    if (shader) {
                        pxr::VtValue val;
                        if (shader.GetInput(pxr::TfToken("diffuseColor")).Get(&val)) {
                            if (val.IsHolding<pxr::GfVec3f>()) {
                                defaultColor = val.Get<pxr::GfVec3f>();
                            }
                        }
                        if (shader.GetInput(pxr::TfToken("opacity")).Get(&val)) {
                            if (val.IsHolding<float>()) {
                                defaultOpacity = val.Get<float>();
                            }
                        }
                    }
                }
            }

            size_t vertexOffset = vertices.size();
            std::vector<std::array<uint32_t, 3>> meshTriangles;

            size_t indexOffset = 0;
            for (size_t f = 0; f < faceVertexCounts.size(); ++f) {
                int vCount = faceVertexCounts[f];
                if (vCount < 3) {
                    indexOffset += vCount;
                    continue;
                }

                for (int v = 1; v < vCount - 1; ++v) {
                    uint32_t i0 = static_cast<uint32_t>(faceVertexIndices[indexOffset + 0]);
                    uint32_t i1 = static_cast<uint32_t>(faceVertexIndices[indexOffset + v]);
                    uint32_t i2 = static_cast<uint32_t>(faceVertexIndices[indexOffset + v + 1]);
                    meshTriangles.push_back({i0, i1, i2});
                }
                indexOffset += vCount;
            }

            for (size_t i = 0; i < points.size(); ++i) {
                pxr::GfVec3f p = points[i];
                pxr::GfVec3d pWorld = transform.Transform(pxr::GfVec3d(p[0], p[1], p[2]));
                vertices.push_back({static_cast<float>(pWorld[0]), static_cast<float>(pWorld[1]), static_cast<float>(pWorld[2])});

                pxr::GfVec3f c = defaultColor;
                if (hasDisplayColor) {
                    c = (displayColors.size() == points.size()) ? displayColors[i] : displayColors[0];
                }
                colors.push_back({c[0], c[1], c[2]});

                float op = defaultOpacity;
                if (hasDisplayOpacity) {
                    op = (displayOpacities.size() == points.size()) ? displayOpacities[i] : displayOpacities[0];
                }
                opacities.push_back(op);
            }

            for (const auto& tri : meshTriangles) {
                indices.push_back(tri[0] + static_cast<uint32_t>(vertexOffset));
                indices.push_back(tri[1] + static_cast<uint32_t>(vertexOffset));
                indices.push_back(tri[2] + static_cast<uint32_t>(vertexOffset));
            }

            foundAny = true;
        }
    }

    if (foundAny && !vertices.empty()) {
        normals.resize(vertices.size(), {0.0f, 0.0f, 0.0f});
        for (size_t i = 0; i < indices.size(); i += 3) {
            uint32_t i0 = indices[i + 0];
            uint32_t i1 = indices[i + 1];
            uint32_t i2 = indices[i + 2];

            float3 v0 = vertices[i0];
            float3 v1 = vertices[i1];
            float3 v2 = vertices[i2];

            float3 edge1 = {v1.x - v0.x, v1.y - v0.y, v1.z - v0.z};
            float3 edge2 = {v2.x - v0.x, v2.y - v0.y, v2.z - v0.z};
            float3 fn = {
                edge1.y * edge2.z - edge1.z * edge2.y,
                edge1.z * edge2.x - edge1.x * edge2.z,
                edge1.x * edge2.y - edge1.y * edge2.x
            };

            normals[i0] = {normals[i0].x + fn.x, normals[i0].y + fn.y, normals[i0].z + fn.z};
            normals[i1] = {normals[i1].x + fn.x, normals[i1].y + fn.y, normals[i1].z + fn.z};
            normals[i2] = {normals[i2].x + fn.x, normals[i2].y + fn.y, normals[i2].z + fn.z};
        }

        for (auto& n : normals) {
            float lenSq = n.x * n.x + n.y * n.y + n.z * n.z;
            if (lenSq > 1e-12f) {
                float invLen = 1.0f / sqrtf(lenSq);
                n = {n.x * invLen, n.y * invLen, n.z * invLen};
            } else {
                n = {0.0f, 1.0f, 0.0f};
            }
        }
    }

    return foundAny;
}

bool loadUsdCamera(const std::string& filename, float3& cam_eye, float3& cam_u, float3& cam_v, float3& cam_w, int width, int height) {
    pxr::UsdStageRefPtr stage = pxr::UsdStage::Open(filename);
    if (!stage) return false;

    pxr::UsdGeomXformCache xformCache;
    pxr::UsdPrim targetPrim;

    for (pxr::UsdPrim prim : stage->Traverse()) {
        if (prim.IsA<pxr::UsdGeomCamera>() && prim.GetName() == "camera1") {
            targetPrim = prim;
            break;
        }
    }
    if (!targetPrim.IsValid()) {
        for (pxr::UsdPrim prim : stage->Traverse()) {
            if (prim.IsA<pxr::UsdGeomCamera>()) {
                targetPrim = prim;
                break;
            }
        }
    }

    if (targetPrim.IsValid()) {
        pxr::GfMatrix4d transform = xformCache.GetLocalToWorldTransform(targetPrim);

        pxr::GfVec3d pos = transform.ExtractTranslation();
        cam_eye = {static_cast<float>(pos[0]), static_cast<float>(pos[1]), static_cast<float>(pos[2])};

        pxr::GfVec3d right = transform.TransformDir(pxr::GfVec3d(1.0, 0.0, 0.0));
        pxr::GfVec3d up    = transform.TransformDir(pxr::GfVec3d(0.0, 1.0, 0.0));
        pxr::GfVec3d view  = transform.TransformDir(pxr::GfVec3d(0.0, 0.0, -1.0));

        float aspect = static_cast<float>(width) / static_cast<float>(height);

        cam_u = {static_cast<float>(right[0]) * aspect, static_cast<float>(right[1]) * aspect, static_cast<float>(right[2]) * aspect};
        cam_v = {static_cast<float>(up[0]), static_cast<float>(up[1]), static_cast<float>(up[2])};
        cam_w = {static_cast<float>(view[0]), static_cast<float>(view[1]), static_cast<float>(view[2])};
        return true;
    }

    return false;
}

int main( int argc, char* argv[] )
{
    std::string usd_filename = (argc > 1) ? argv[1] : "sphere.usd";

    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW\n";
        return 1;
    }

    const int width = 768;
    const int height = 768;
    GLFWwindow* window = glfwCreateWindow(width, height, "OptiX Live Ray Traced Viewer", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to open GLFW window\n";
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);

    CUDA_CHECK( cudaFree( 0 ) );
    OPTIX_CHECK( optixInit() );

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = [](unsigned int level, const char* tag, const char* message, void* cbdata) {
        fprintf(stderr, "[OptiX Log][%2d][%s]: %s\n", level, tag, message);
    };
    options.logCallbackLevel = 4;

    OptixDeviceContext context = nullptr;
    CUcontext cu_ctx = 0;
    OPTIX_CHECK( optixDeviceContextCreate( cu_ctx, &options, &context ) );

    std::vector<float3> vertices;
    std::vector<float3> normals;
    std::vector<float3> colors;
    std::vector<float> opacities;
    std::vector<uint32_t> indices;
    if (!loadUsdMesh(usd_filename, vertices, normals, colors, opacities, indices)) {
        std::cerr << "Aborting due to USD loading failure.\n";
        glfwTerminate();
        return 1;
    }

    CUdeviceptr d_vertices = 0;
    CUDA_CHECK( cudaMalloc( (void**)&d_vertices, vertices.size() * sizeof( float3 ) ) );
    CUDA_CHECK( cudaMemcpy( (void*)d_vertices, vertices.data(), vertices.size() * sizeof( float3 ), cudaMemcpyHostToDevice ) );

    CUdeviceptr d_normals = 0;
    CUDA_CHECK( cudaMalloc( (void**)&d_normals, normals.size() * sizeof( float3 ) ) );
    CUDA_CHECK( cudaMemcpy( (void*)d_normals, normals.data(), normals.size() * sizeof( float3 ), cudaMemcpyHostToDevice ) );

    CUdeviceptr d_colors = 0;
    CUDA_CHECK( cudaMalloc( (void**)&d_colors, colors.size() * sizeof( float3 ) ) );
    CUDA_CHECK( cudaMemcpy( (void*)d_colors, colors.data(), colors.size() * sizeof( float3 ), cudaMemcpyHostToDevice ) );

    CUdeviceptr d_opacities = 0;
    CUDA_CHECK( cudaMalloc( (void**)&d_opacities, opacities.size() * sizeof( float ) ) );
    CUDA_CHECK( cudaMemcpy( (void*)d_opacities, opacities.data(), opacities.size() * sizeof( float ), cudaMemcpyHostToDevice ) );

    CUdeviceptr d_indices = 0;
    CUDA_CHECK( cudaMalloc( (void**)&d_indices, indices.size() * sizeof( uint32_t ) ) );
    CUDA_CHECK( cudaMemcpy( (void*)d_indices, indices.data(), indices.size() * sizeof( uint32_t ), cudaMemcpyHostToDevice ) );

    OptixBuildInput triangle_input = {};
    uint32_t triangle_input_flags = OPTIX_GEOMETRY_FLAG_NONE;

    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.vertexStrideInBytes = sizeof( float3 );
    triangle_input.triangleArray.numVertices = static_cast<uint32_t>( vertices.size() );
    triangle_input.triangleArray.vertexBuffers = &d_vertices;

    triangle_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangle_input.triangleArray.indexStrideInBytes = 3 * sizeof( uint32_t );
    triangle_input.triangleArray.numIndexTriplets = static_cast<uint32_t>( indices.size() / 3 );
    triangle_input.triangleArray.indexBuffer = d_indices;

    triangle_input.triangleArray.flags = &triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK( optixAccelComputeMemoryUsage( context, &accel_options, &triangle_input, 1, &gas_buffer_sizes ) );

    CUdeviceptr d_temp_buffer_gas = 0;
    CUDA_CHECK( cudaMalloc( (void**)&d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes ) );

    CUdeviceptr d_buffer_temp_output_gas_and_cup = 0;
    CUDA_CHECK( cudaMalloc( (void**)&d_buffer_temp_output_gas_and_cup, gas_buffer_sizes.outputSizeInBytes ) );

    OptixTraversableHandle gas_handle = 0;
    OPTIX_CHECK( optixAccelBuild(
                context, 0, &accel_options, &triangle_input, 1,
                d_temp_buffer_gas, gas_buffer_sizes.tempSizeInBytes,
                d_buffer_temp_output_gas_and_cup, gas_buffer_sizes.outputSizeInBytes,
                &gas_handle, nullptr, 0
                ) );
    CUDA_CHECK( cudaFree( (void*)d_temp_buffer_gas ) );

    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;

    OptixPipelineCompileOptions pipeline_compile_options = {};
    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.numPayloadValues = 4;
    pipeline_compile_options.numAttributeValues = 2;
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    std::string ptx_code = readPTXFile("optixTriangle.ptx");

    OptixModule module = nullptr;
    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK( optixModuleCreate(
                context, &module_compile_options, &pipeline_compile_options,
                ptx_code.c_str(), ptx_code.size(), log, &sizeof_log, &module
                ) );

    OptixProgramGroupOptions pg_options = {};

    OptixProgramGroupDesc rg_prog_group_desc = {};
    rg_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rg_prog_group_desc.raygen.module = module;
    rg_prog_group_desc.raygen.entryFunctionName = "__raygen__rg";
    OptixProgramGroup raygen_prog_group = nullptr;
    OPTIX_CHECK( optixProgramGroupCreate( context, &rg_prog_group_desc, 1, &pg_options, log, &sizeof_log, &raygen_prog_group ) );

    OptixProgramGroupDesc ms_prog_group_desc = {};
    ms_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    ms_prog_group_desc.miss.module = module;
    ms_prog_group_desc.miss.entryFunctionName = "__miss__ms";
    OptixProgramGroup miss_prog_group = nullptr;
    OPTIX_CHECK( optixProgramGroupCreate( context, &ms_prog_group_desc, 1, &pg_options, log, &sizeof_log, &miss_prog_group ) );

    OptixProgramGroupDesc hit_prog_group_desc = {};
    hit_prog_group_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hit_prog_group_desc.hitgroup.moduleCH = module;
    hit_prog_group_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    OptixProgramGroup hitgroup_prog_group = nullptr;
    OPTIX_CHECK( optixProgramGroupCreate( context, &hit_prog_group_desc, 1, &pg_options, log, &sizeof_log, &hitgroup_prog_group ) );

    OptixProgramGroup program_groups[] = { raygen_prog_group, miss_prog_group, hitgroup_prog_group };
    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 4;

    OptixPipeline pipeline = nullptr;
    OPTIX_CHECK( optixPipelineCreate(
                context, &pipeline_compile_options, &pipeline_link_options,
                program_groups, sizeof( program_groups ) / sizeof( program_groups[0] ),
                log, &sizeof_log, &pipeline
                ) );

    OPTIX_CHECK( optixPipelineSetStackSize( pipeline, 32 * 1024, 32 * 1024, 32 * 1024, 1 ) );

    CUdeviceptr d_raygen_record;
    CUDA_CHECK( cudaMalloc( (void**)&d_raygen_record, sizeof( RayGenRecord ) ) );
    RayGenRecord rg_sbt = {};
    OPTIX_CHECK( optixSbtRecordPackHeader( raygen_prog_group, &rg_sbt ) );
    CUDA_CHECK( cudaMemcpy( (void*)d_raygen_record, &rg_sbt, sizeof( RayGenRecord ), cudaMemcpyHostToDevice ) );

    CUdeviceptr d_miss_record;
    CUDA_CHECK( cudaMalloc( (void**)&d_miss_record, sizeof( MissRecord ) ) );
    MissRecord ms_sbt = {};
    OPTIX_CHECK( optixSbtRecordPackHeader( miss_prog_group, &ms_sbt ) );
    CUDA_CHECK( cudaMemcpy( (void*)d_miss_record, &ms_sbt, sizeof( MissRecord ), cudaMemcpyHostToDevice ) );

    CUdeviceptr d_hitgroup_record;
    CUDA_CHECK( cudaMalloc( (void**)&d_hitgroup_record, sizeof( HitGroupRecord ) ) );
    HitGroupRecord hg_sbt = {};
    OPTIX_CHECK( optixSbtRecordPackHeader( hitgroup_prog_group, &hg_sbt ) );
    CUDA_CHECK( cudaMemcpy( (void*)d_hitgroup_record, &hg_sbt, sizeof( HitGroupRecord ), cudaMemcpyHostToDevice ) );

    OptixShaderBindingTable sbt = {};
    sbt.raygenRecord = d_raygen_record;
    sbt.missRecordBase = d_miss_record;
    sbt.missRecordStrideInBytes = sizeof( MissRecord );
    sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = d_hitgroup_record;
    sbt.hitgroupRecordStrideInBytes = sizeof( HitGroupRecord );
    sbt.hitgroupRecordCount = 1;

    std::vector<uchar4> h_image( width * height );
    uchar4* d_image = nullptr;
    CUDA_CHECK( cudaMalloc( (void**)&d_image, width * height * sizeof( uchar4 ) ) );

    Params params;
    params.image = d_image;
    params.image_width = width;
    params.image_height = height;
    params.handle = gas_handle;
    params.vertices = reinterpret_cast<float3*>(d_vertices);
    params.colors = reinterpret_cast<float3*>(d_colors);
    params.normals = reinterpret_cast<float3*>(d_normals);
    params.opacities = reinterpret_cast<float*>(d_opacities);
    params.indices = reinterpret_cast<uint32_t*>(d_indices);

    if (!loadUsdCamera(usd_filename, params.cam_eye, params.cam_u, params.cam_v, params.cam_w, width, height)) {
        params.cam_eye = { 0.0f, 0.0f, 3.5f };
        params.cam_u = { 1.0f, 0.0f, 0.0f };
        params.cam_v = { 0.0f, 1.0f, 0.0f };
        params.cam_w = { 0.0f, 0.0f, -1.0f };
    }

    params.light_pos = params.cam_eye + make_float3(2.0f, 3.0f, 2.0f);

    Params* d_params;
    CUDA_CHECK( cudaMalloc( (void**)&d_params, sizeof( Params ) ) );
    CUDA_CHECK( cudaMemcpy( d_params, &params, sizeof( Params ), cudaMemcpyHostToDevice ) );

    // Setup OpenGL Texture for Display
    GLuint texOutput;
    glGenTextures(1, &texOutput);
    glBindTexture(GL_TEXTURE_2D, texOutput);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);

    std::cout << "Starting live render loop in window. Close window to exit.\n";

    // Main Render & Display Loop
    while (!glfwWindowShouldClose(window)) {
        // Execute OptiX Trace
        OPTIX_CHECK( optixLaunch( pipeline, 0, (CUdeviceptr)d_params, sizeof( Params ), &sbt, width, height, 1 ) );
        CUDA_CHECK( cudaDeviceSynchronize() );

        // Copy Image to Host
        CUDA_CHECK( cudaMemcpy( h_image.data(), d_image, width * height * sizeof( uchar4 ), cudaMemcpyDeviceToHost ) );

        // Upload to OpenGL Texture
        glBindTexture(GL_TEXTURE_2D, texOutput);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, h_image.data());

        // Render Full-Screen Quad
        glClear(GL_COLOR_BUFFER_BIT);
        glEnable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, texOutput);

        glBegin(GL_QUADS);
        glTexCoord2f(0.0f, 1.0f); glVertex2f(-1.0f, -1.0f);
        glTexCoord2f(1.0f, 1.0f); glVertex2f( 1.0f, -1.0f);
        glTexCoord2f(1.0f, 0.0f); glVertex2f( 1.0f,  1.0f);
        glTexCoord2f(0.0f, 0.0f); glVertex2f(-1.0f,  1.0f);
        glEnd();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // Cleanup
    cudaFree( (void*)d_vertices );
    cudaFree( (void*)d_normals );
    cudaFree( (void*)d_colors );
    cudaFree( (void*)d_opacities );
    cudaFree( (void*)d_indices );
    cudaFree( (void*)d_buffer_temp_output_gas_and_cup );
    cudaFree( (void*)d_raygen_record );
    cudaFree( (void*)d_miss_record );
    cudaFree( (void*)d_hitgroup_record );
    cudaFree( (void*)d_image );
    cudaFree( (void*)d_params );
    optixPipelineDestroy( pipeline );
    optixProgramGroupDestroy( raygen_prog_group );
    optixProgramGroupDestroy( miss_prog_group );
    optixProgramGroupDestroy( hitgroup_prog_group );
    optixModuleDestroy( module );
    optixDeviceContextDestroy( context );

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}

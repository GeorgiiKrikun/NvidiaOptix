#include <optix.h>
#include <optix_stubs.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>

#include "common.h"

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " code=" << error << " \"" << cudaGetErrorString(error) << "\"" << std::endl; \
            exit(1); \
        } \
    } while (0)

#define OPTIX_CHECK(call) \
    do { \
        OptixResult res = call; \
        if (res != OPTIX_SUCCESS) { \
            std::cerr << "OptiX error at " << __FILE__ << ":" << __LINE__ << " code=" << res << std::endl; \
            exit(1); \
        } \
    } while (0)

template <typename T>
struct SbtRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData>   RayGenSbtRecord;
typedef SbtRecord<MissData>     MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;

void initOptix(OptixDeviceContext& context)
{
    // Initialize CUDA
    CUDA_CHECK(cudaFree(0));

    // Initialize OptiX
    OPTIX_CHECK(optixInit());

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = [](unsigned int level, const char* tag, const char* cbdata, void* ) {
        std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << cbdata << std::endl;
    };
    options.logCallbackLevel = 4;

    CUcontext cuCtx = 0; // current context
    OPTIX_CHECK(optixDeviceContextCreate(cuCtx, &options, &context));
}

void createPipeline(OptixDeviceContext context, OptixPipeline& pipeline, OptixModule& module,
                    OptixProgramGroup& raygen_pg, OptixProgramGroup& miss_pg, OptixProgramGroup& hitgroup_pg)
{
    OptixModuleCompileOptions module_compile_options = {};

    OptixPipelineCompileOptions pipeline_compile_options = {};
    pipeline_compile_options.usesMotionBlur = false;
    pipeline_compile_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipeline_compile_options.numPayloadValues = 3;
    pipeline_compile_options.numAttributeValues = 2;
    pipeline_compile_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compile_options.pipelineLaunchParamsVariableName = "params";

    // Load PTX from file
    std::ifstream ptx_file("device_programs.ptx");
    if (!ptx_file.is_open()) {
        std::cerr << "Failed to open device_programs.ptx" << std::endl;
        exit(1);
    }
    std::string ptx((std::istreambuf_iterator<char>(ptx_file)), std::istreambuf_iterator<char>());

    char log[2048];
    size_t sizeof_log = sizeof(log);
    OPTIX_CHECK(optixModuleCreateFromPTX(
        context,
        &module_compile_options,
        &pipeline_compile_options,
        ptx.c_str(),
        ptx.size(),
        log,
        &sizeof_log,
        &module
    ));

    OptixProgramGroupOptions program_group_options = {};

    OptixProgramGroupDesc raygen_pg_desc = {};
    raygen_pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    raygen_pg_desc.raygen.module = module;
    raygen_pg_desc.raygen.entryFunctionName = "__raygen__rg";
    OPTIX_CHECK(optixProgramGroupCreate(context, &raygen_pg_desc, 1, &program_group_options, log, &sizeof_log, &raygen_pg));

    OptixProgramGroupDesc miss_pg_desc = {};
    miss_pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    miss_pg_desc.miss.module = module;
    miss_pg_desc.miss.entryFunctionName = "__miss__ms";
    OPTIX_CHECK(optixProgramGroupCreate(context, &miss_pg_desc, 1, &program_group_options, log, &sizeof_log, &miss_pg));

    OptixProgramGroupDesc hitgroup_pg_desc = {};
    hitgroup_pg_desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hitgroup_pg_desc.hitgroup.moduleCH = module;
    hitgroup_pg_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    OPTIX_CHECK(optixProgramGroupCreate(context, &hitgroup_pg_desc, 1, &program_group_options, log, &sizeof_log, &hitgroup_pg));

    OptixProgramGroup program_groups[] = { raygen_pg, miss_pg, hitgroup_pg };
    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 1;
    pipeline_link_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    OPTIX_CHECK(optixPipelineCreate(
        context,
        &pipeline_compile_options,
        &pipeline_link_options,
        program_groups,
        sizeof(program_groups) / sizeof(program_groups[0]),
        log,
        &sizeof_log,
        &pipeline
    ));
}

void buildAccel(OptixDeviceContext context, OptixTraversableHandle& gas_handle, CUdeviceptr& d_gas_output_buffer)
{
    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_NONE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    const std::vector<float3> vertices = {
        { -0.5f, -0.5f, 0.0f },
        {  0.5f, -0.5f, 0.0f },
        {  0.0f,  0.5f, 0.0f }
    };

    const size_t vertices_size = sizeof(float3) * vertices.size();
    CUdeviceptr d_vertices;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_vertices), vertices_size));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_vertices), vertices.data(), vertices_size, cudaMemcpyHostToDevice));

    const uint32_t triangle_input_flags[1] = { OPTIX_GEOMETRY_FLAG_NONE };
    OptixBuildInput triangle_input = {};
    triangle_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    triangle_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangle_input.triangleArray.numVertices = static_cast<uint32_t>(vertices.size());
    triangle_input.triangleArray.vertexBuffers = &d_vertices;
    triangle_input.triangleArray.flags = triangle_input_flags;
    triangle_input.triangleArray.numSbtRecords = 1;

    OptixAccelBufferSizes gas_buffer_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context, &accel_options, &triangle_input, 1, &gas_buffer_sizes));

    CUdeviceptr d_temp_buffer_gas;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_temp_buffer_gas), gas_buffer_sizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_gas_output_buffer), gas_buffer_sizes.outputSizeInBytes));

    OPTIX_CHECK(optixAccelBuild(
        context,
        0, // CUDA stream
        &accel_options,
        &triangle_input,
        1, // num build inputs
        d_temp_buffer_gas,
        gas_buffer_sizes.tempSizeInBytes,
        d_gas_output_buffer,
        gas_buffer_sizes.outputSizeInBytes,
        &gas_handle,
        nullptr, // emitted property list
        0        // num emitted properties
    ));

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_temp_buffer_gas)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_vertices)));
}

void setupSBT(OptixShaderBindingTable& sbt, OptixProgramGroup raygen_pg, OptixProgramGroup miss_pg, OptixProgramGroup hitgroup_pg)
{
    CUdeviceptr d_raygen_record;
    const size_t raygen_record_size = sizeof(RayGenSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), raygen_record_size));
    RayGenSbtRecord rg_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_pg, &rg_sbt));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_raygen_record), &rg_sbt, raygen_record_size, cudaMemcpyHostToDevice));

    CUdeviceptr d_miss_record;
    const size_t miss_record_size = sizeof(MissSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_record), miss_record_size));
    MissSbtRecord ms_sbt;
    ms_sbt.data.bg_color = { 0.3f, 0.1f, 0.2f };
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_pg, &ms_sbt));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_miss_record), &ms_sbt, miss_record_size, cudaMemcpyHostToDevice));

    CUdeviceptr d_hitgroup_record;
    const size_t hitgroup_record_size = sizeof(HitGroupSbtRecord);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitgroup_record), hitgroup_record_size));
    HitGroupSbtRecord hg_sbt;
    OPTIX_CHECK(optixSbtRecordPackHeader(hitgroup_pg, &hg_sbt));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hitgroup_record), &hg_sbt, hitgroup_record_size, cudaMemcpyHostToDevice));

    sbt.raygenRecord = d_raygen_record;
    sbt.missRecordBase = d_miss_record;
    sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = d_hitgroup_record;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    sbt.hitgroupRecordCount = 1;
}

int main()
{
    OptixDeviceContext context;
    initOptix(context);

    OptixPipeline pipeline;
    OptixModule module;
    OptixProgramGroup raygen_pg, miss_pg, hitgroup_pg;
    createPipeline(context, pipeline, module, raygen_pg, miss_pg, hitgroup_pg);

    OptixTraversableHandle gas_handle;
    CUdeviceptr d_gas_output_buffer;
    buildAccel(context, gas_handle, d_gas_output_buffer);

    OptixShaderBindingTable sbt = {};
    setupSBT(sbt, raygen_pg, miss_pg, hitgroup_pg);

    // Final Launch
    const int width = 512;
    const int height = 512;

    uchar4* h_image = new uchar4[width * height];
    uchar4* d_image;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_image), width * height * sizeof(uchar4)));

    Params params;
    params.image = d_image;
    params.image_width = width;
    params.image_height = height;
    params.cam_eye = { 0.0f, 0.0f, 2.0f };
    params.cam_u = { 1.0f, 0.0f, 0.0f };
    params.cam_v = { 0.0f, 1.0f, 0.0f };
    params.cam_w = { 0.0f, 0.0f, -1.0f };
    params.handle = gas_handle;

    CUdeviceptr d_params;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(Params)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_params), &params, sizeof(Params), cudaMemcpyHostToDevice));

    OPTIX_CHECK(optixLaunch(pipeline, 0, d_params, sizeof(Params), &sbt, width, height, 1));

    CUDA_CHECK(cudaMemcpy(h_image, d_image, width * height * sizeof(uchar4), cudaMemcpyDeviceToHost));

    // Save image as PPM
    std::ofstream out("output.ppm");
    out << "P3\n" << width << " " << height << "\n255\n";
    for (int i = 0; i < width * height; ++i) {
        out << (int)h_image[i].x << " " << (int)h_image[i].y << " " << (int)h_image[i].z << "\n";
    }

    std::cout << "OptiX launch complete. Output saved to output.ppm" << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_params)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_image)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(d_gas_output_buffer)));
    OPTIX_CHECK(optixPipelineDestroy(pipeline));
    OPTIX_CHECK(optixProgramGroupDestroy(raygen_pg));
    OPTIX_CHECK(optixProgramGroupDestroy(miss_pg));
    OPTIX_CHECK(optixProgramGroupDestroy(hitgroup_pg));
    OPTIX_CHECK(optixModuleDestroy(module));
    OPTIX_CHECK(optixDeviceContextDestroy(context));

    delete[] h_image;

    return 0;
}

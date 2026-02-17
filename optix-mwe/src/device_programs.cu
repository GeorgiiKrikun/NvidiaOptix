#include <optix.h>
#include "common.h"
#include <cuda_runtime.h>

extern "C" {
    __constant__ Params params;
}

static __forceinline__ __device__ void setPayload( float3 p )
{
    optixSetPayload_0( __float_as_int( p.x ) );
    optixSetPayload_1( __float_as_int( p.y ) );
    optixSetPayload_2( __float_as_int( p.z ) );
}

static __forceinline__ __device__ float3 getPayload()
{
    return make_float3(
        __int_as_float( optixGetPayload_0() ),
        __int_as_float( optixGetPayload_1() ),
        __int_as_float( optixGetPayload_2() )
    );
}

extern "C" __global__ void __raygen__rg()
{
    // Lookup our location within the launch grid
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();

    // Map our launch idx to a screen location and create a ray from
    // the camera location through the screen
    float2 d = make_float2(idx.x, idx.y) / make_float2(dim.x, dim.y) * 2.f - 1.f;
    float3 ray_origin = params.cam_eye;
    float3 ray_direction = normalize(d.x * params.cam_u + d.y * params.cam_v + params.cam_w);

    // Trace the ray against our scene hierarchy
    unsigned int p0, p1, p2;
    optixTrace(
        params.handle,
        ray_origin,
        ray_direction,
        0.0f,                // Min intersection distance
        1e16f,               // Max intersection distance
        0.0f,                // ray-time -- used for motion blur
        OptixVisibilityMask( 255 ), // Specify always visible
        OPTIX_RAY_FLAG_NONE,
        0,                   // SBT offset
        0,                   // SBT stride
        0,                   // missSBTIndex
        p0, p1, p2 );

    float3 result = make_float3(__int_as_float(p0), __int_as_float(p1), __int_as_float(p2));

    // Record results in our output raster
    params.image[idx.y * params.image_width + idx.x] = make_uchar4(
        static_cast<unsigned char>(result.x * 255.5f),
        static_cast<unsigned char>(result.y * 255.5f),
        static_cast<unsigned char>(result.z * 255.5f),
        255
    );
}

extern "C" __global__ void __miss__ms()
{
    MissData* miss_data = reinterpret_cast<MissData*>( optixGetSbtDataPointer() );
    setPayload( miss_data->bg_color );
}

extern "C" __global__ void __closesthit__ch()
{
    // When built-in triangle intersection is used, a number of fundamental
    // attributes are provided by the OptiX API, including barycentric
    // coordinates.
    const float2 barycentrics = optixGetTriangleBarycentrics();

    // Convert to color and assign to our payload outputs.
    const float3 c = make_float3( barycentrics.x, barycentrics.y, 1.0f - barycentrics.x - barycentrics.y );
    setPayload( c );
}

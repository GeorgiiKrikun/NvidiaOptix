#pragma once

#include <cuda_runtime.h>
#include <optix.h>

struct Params
{
    uchar4*                image;
    unsigned int           image_width;
    unsigned int           image_height;
    float3                 cam_eye;
    float3                 cam_u, cam_v, cam_w;
    OptixTraversableHandle handle;
};

struct RayGenData
{
};

struct MissData
{
    float3 bg_color;
};

struct HitGroupData
{
};

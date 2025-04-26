#pragma once
#include <cuda_runtime.h>

// CUDA包装函数声明
void launchInitializeFields(float* velocity_x, float* velocity_y, float* pressure, float* temperature,
                          int width, int height, cudaStream_t stream);

void launchComputeVelocityField(float* velocity_x, float* velocity_y, float* pressure, float* temperature,
                               int width, int height, float dt, cudaStream_t stream);

void launchComputePressureField(float* pressure, float* velocity_x, float* velocity_y,
                               int width, int height, float dt, cudaStream_t stream); 
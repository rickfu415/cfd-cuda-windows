#include "cuda_wrapper.h"
#include <cuda_runtime.h>

// CUDA kernel implementations
__global__ void initializeFields(float* velocity_x, float* velocity_y, float* pressure, float* temperature,
                               int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    // Set initial conditions
    velocity_x[idx] = 0.0f;
    velocity_y[idx] = 0.0f;
    pressure[idx] = 0.0f;
    temperature[idx] = 300.0f;  // Initial temperature 300K
}

__global__ void computeVelocityField(float* velocity_x, float* velocity_y, float* pressure, float* temperature,
                                   int width, int height, float dt) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    // Simple convection-diffusion equation
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        // Calculate pressure gradient
        float dp_dx = (pressure[idx + 1] - pressure[idx - 1]) / 2.0f;
        float dp_dy = (pressure[(y + 1) * width + x] - pressure[(y - 1) * width + x]) / 2.0f;
        
        // Update velocity
        velocity_x[idx] -= dt * dp_dx;
        velocity_y[idx] -= dt * dp_dy;
    }
}

__global__ void computePressureField(float* pressure, float* velocity_x, float* velocity_y,
                                   int width, int height, float dt) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    // Simple pressure Poisson equation solver
    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        float div = (velocity_x[idx + 1] - velocity_x[idx - 1]) / 2.0f +
                   (velocity_y[(y + 1) * width + x] - velocity_y[(y - 1) * width + x]) / 2.0f;
        
        pressure[idx] += dt * div;
    }
}

// CUDA wrapper function implementations
void launchInitializeFields(float* velocity_x, float* velocity_y, float* pressure, float* temperature,
                          int width, int height, cudaStream_t stream) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    initializeFields<<<gridSize, blockSize, 0, stream>>>(
        velocity_x, velocity_y, pressure, temperature,
        width, height
    );
}

void launchComputeVelocityField(float* velocity_x, float* velocity_y, float* pressure, float* temperature,
                               int width, int height, float dt, cudaStream_t stream) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    computeVelocityField<<<gridSize, blockSize, 0, stream>>>(
        velocity_x, velocity_y, pressure, temperature,
        width, height, dt
    );
}

void launchComputePressureField(float* pressure, float* velocity_x, float* velocity_y,
                               int width, int height, float dt, cudaStream_t stream) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    computePressureField<<<gridSize, blockSize, 0, stream>>>(
        pressure, velocity_x, velocity_y,
        width, height, dt
    );
} 
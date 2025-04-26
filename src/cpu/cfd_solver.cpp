#include "cfd_solver.h"
#include "cuda_wrapper.h"
#include <cuda_runtime.h>
#include <iostream>

// 声明CUDA核函数
extern "C" {
    void computeVelocityField(float* velocity_x, float* velocity_y, float* pressure, float* temperature,
                            int width, int height, float dt);
    void computePressureField(float* pressure, float* velocity_x, float* velocity_y,
                            int width, int height, float dt);
    void initializeFields(float* velocity_x, float* velocity_y, float* pressure, float* temperature,
                         int width, int height);
}

CFDSolver::CFDSolver(int width, int height)
    : width_(width), height_(height) {
    // 分配CPU内存
    h_velocity_x_.resize(width * height);
    h_velocity_y_.resize(width * height);
    h_pressure_.resize(width * height);
    h_temperature_.resize(width * height);
    
    // 分配GPU内存
    cudaMalloc(&d_velocity_x_, width * height * sizeof(float));
    cudaMalloc(&d_velocity_y_, width * height * sizeof(float));
    cudaMalloc(&d_pressure_, width * height * sizeof(float));
    cudaMalloc(&d_temperature_, width * height * sizeof(float));
    
    // 创建CUDA流
    cudaStreamCreate(&stream_);
}

CFDSolver::~CFDSolver() {
    // 释放GPU内存
    cudaFree(d_velocity_x_);
    cudaFree(d_velocity_y_);
    cudaFree(d_pressure_);
    cudaFree(d_temperature_);
    
    // 销毁CUDA流
    cudaStreamDestroy(stream_);
}

void CFDSolver::initialize() {
    // 初始化场
    launchInitializeFields(
        d_velocity_x_, d_velocity_y_, d_pressure_, d_temperature_,
        width_, height_, stream_
    );
    
    // 同步等待初始化完成
    cudaStreamSynchronize(stream_);
}

void CFDSolver::step() {
    float dt = 0.01f;  // 时间步长
    
    // 计算速度场
    launchComputeVelocityField(
        d_velocity_x_, d_velocity_y_, d_pressure_, d_temperature_,
        width_, height_, dt, stream_
    );
    
    // 计算压力场
    launchComputePressureField(
        d_pressure_, d_velocity_x_, d_velocity_y_,
        width_, height_, dt, stream_
    );
    
    // 同步等待计算完成
    cudaStreamSynchronize(stream_);
}

const float* CFDSolver::getVelocityField() const {
    // 将速度场从GPU复制到CPU
    cudaMemcpyAsync(const_cast<float*>(h_velocity_x_.data()), d_velocity_x_,
                    width_ * height_ * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    cudaMemcpyAsync(const_cast<float*>(h_velocity_y_.data()), d_velocity_y_,
                    width_ * height_ * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    
    // 同步等待复制完成
    cudaStreamSynchronize(stream_);
    
    return h_velocity_x_.data();
}

const float* CFDSolver::getPressureField() const {
    // 将压力场从GPU复制到CPU
    cudaMemcpyAsync(const_cast<float*>(h_pressure_.data()), d_pressure_,
                    width_ * height_ * sizeof(float),
                    cudaMemcpyDeviceToHost, stream_);
    
    // 同步等待复制完成
    cudaStreamSynchronize(stream_);
    
    return h_pressure_.data();
} 
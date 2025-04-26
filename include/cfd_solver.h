#pragma once

#include <vector>
#include <cuda_runtime.h>

class CFDSolver {
public:
    CFDSolver(int width, int height);
    ~CFDSolver();

    // 初始化求解器
    void initialize();
    
    // 运行一个时间步
    void step();
    
    // 获取当前状态
    const float* getVelocityField() const;
    const float* getPressureField() const;

private:
    // 网格尺寸
    int width_;
    int height_;
    
    // GPU内存
    float* d_velocity_x_;
    float* d_velocity_y_;
    float* d_pressure_;
    float* d_temperature_;
    
    // CPU内存（用于可视化）
    std::vector<float> h_velocity_x_;
    std::vector<float> h_velocity_y_;
    std::vector<float> h_pressure_;
    std::vector<float> h_temperature_;

    // CUDA流
    cudaStream_t stream_;
}; 
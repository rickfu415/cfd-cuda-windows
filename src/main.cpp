#include "cfd_solver.h"
#include <iostream>
#include <chrono>
#include <thread>

int main() {
    // 创建CFD求解器实例
    const int width = 256;
    const int height = 256;
    CFDSolver solver(width, height);
    
    // 初始化求解器
    solver.initialize();
    
    // 运行模拟
    const int num_steps = 1000;
    for (int step = 0; step < num_steps; ++step) {
        // 记录开始时间
        auto start = std::chrono::high_resolution_clock::now();
        
        // 执行一个时间步
        solver.step();
        
        // 记录结束时间
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        
        // 打印进度
        if (step % 100 == 0) {
            std::cout << "Step " << step << "/" << num_steps 
                      << " completed in " << duration.count() / 1000.0 << "ms" << std::endl;
        }
        
        // 获取当前状态（用于可视化）
        const float* velocity = solver.getVelocityField();
        const float* pressure = solver.getPressureField();
        
        // 这里可以添加可视化代码
        // ...
    }
    
    return 0;
} 
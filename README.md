# GPU-Accelerated CFD Solver Demo

这是一个使用 CUDA 加速的 2D CFD 求解器演示项目。该项目实现了基本的 Navier-Stokes 方程求解，并使用 GPU 进行加速计算。

## 项目结构

```
.
├── src/                    # 源代码目录
│   ├── cuda/              # CUDA加速核心代码
│   └── cpu/               # CPU端代码
├── include/               # 头文件
├── build/                 # 构建目录
├── CMakeLists.txt        # CMake构建配置
└── README.md             # 项目说明文档
```

## 依赖要求

- CUDA Toolkit (>= 11.0)
- CMake (>= 3.10)
- C++编译器 (支持 C++17)
- OpenGL (用于可视化)

## 构建方法

```bash
mkdir build
cd build
cmake ..
make
```

## 运行方法

```bash
./cfd_solver
```

## 功能特点

- 2D Navier-Stokes 方程求解
- GPU 加速计算
- 实时可视化
- 可配置的求解参数

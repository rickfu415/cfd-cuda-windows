cmake_minimum_required(VERSION 3.10)
project(GPU_CFD_Solver CUDA CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 查找CUDA
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})

# 添加源文件
set(SOURCES
    src/cpu/cfd_solver.cpp
    src/main.cpp
)

set(CUDA_SOURCES
    src/cuda/cuda_wrapper.cu
)

# 添加头文件目录
include_directories(include)

# 创建可执行文件
add_executable(cfd_solver ${SOURCES} ${CUDA_SOURCES})

# 设置CUDA架构
set_target_properties(cfd_solver PROPERTIES
    CUDA_ARCHITECTURES "75"
    CUDA_SEPARABLE_COMPILATION ON
)

# 链接必要的库
target_link_libraries(cfd_solver
    ${CUDA_LIBRARIES}
) 
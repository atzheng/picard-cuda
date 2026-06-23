cmake_minimum_required(VERSION 3.18)
project(cuda_sim LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
# compute_70 (Volta) was removed in CUDA 13; keep architectures supported by the toolkit
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)

# Find required packages
find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(cuda_sim
    src/main.cpp
    src/data_loader.cpp
    src/algorithm.cu
)

target_include_directories(cuda_sim PRIVATE include)
target_link_libraries(cuda_sim PRIVATE CUDA::curand OpenMP::OpenMP_CXX)

# Enable OpenMP in the host portions of the CUDA translation unit (the Picard
# host orchestration is the bottleneck and is parallelized across CPU cores).
target_compile_options(cuda_sim PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)

# Enable separable compilation for device-side linking
set_target_properties(cuda_sim PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

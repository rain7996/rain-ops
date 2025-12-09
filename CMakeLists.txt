cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(CudaProject LANGUAGES CXX CUDA)

# set environments
set(PROG "matmul" CACHE STRING "My Cmake Workspace")
set(CMAKE_OUT_DIR "build")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build/lib)

# cpp standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# cuda archetecture
set(CMAKE_CUDA_ARCHITECTURES 90)  # 例如: Turing(75), Ampere(80), Ada(86)

# find packages
find_package(CUDAToolkit REQUIRED)


file(GLOB_RECURSE CUDA_SOURCES 
    "${PROG}/*.cu"
    "${PROG}/*.cc"
)

# create the executable program
add_executable(${PROG}.out
    ${CUDA_SOURCES}
)

message(STATUS "FOUND CUDA FILES:\n ${CUDA_SOURCES}")
get_target_property(TARGET_OUTPUT_DIR ${PROG}.out RUNTIME_OUTPUT_DIRECTORY)
message(STATUS "Target output directory: ${TARGET_OUTPUT_DIR}")

# set link libraries
target_link_libraries(${PROG}.out
    CUDA::cudart
    CUDA::cublas
)

# set compile options
target_compile_options(${PROG}.out PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--default-stream per-thread>
)

# set include directories
target_include_directories(${PROG}.out PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)
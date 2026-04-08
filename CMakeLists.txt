cmake_minimum_required(VERSION 3.18)
project(parallelink LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# -------------------------------------------------------------------
#  Paths
# -------------------------------------------------------------------
set(FIO_DIR    ${CMAKE_SOURCE_DIR}/extern/fio)
set(LIBNVM_DIR ${CMAKE_SOURCE_DIR}/extern/bam)

# -------------------------------------------------------------------
#  fio must be built separately (./configure && make in extern/fio).
#  We only need its headers for the engine plugin.
# -------------------------------------------------------------------
set(FIO_INCLUDE_DIRS
    ${FIO_DIR}
    ${FIO_DIR}/arch
    ${FIO_DIR}/os
)

# -------------------------------------------------------------------
#  libnvm headers (GPU-direct NVMe userspace driver)
# -------------------------------------------------------------------
set(LIBNVM_INCLUDE_DIRS
    ${LIBNVM_DIR}/include
    ${LIBNVM_DIR}/src
)

# -------------------------------------------------------------------
#  CUDA
# -------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# -------------------------------------------------------------------
#  Engine shared library: parallelink.so
#
#  fio loads external engines via dlopen("parallelink.so").
# -------------------------------------------------------------------
add_library(parallelink SHARED
    src/gpu-engine.c
    src/gpu_worker.cu
)

target_include_directories(parallelink PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${FIO_INCLUDE_DIRS}
    ${LIBNVM_INCLUDE_DIRS}
)

target_link_libraries(parallelink PRIVATE
    CUDA::cudart
)

# fio expects the engine symbol to be visible
set_target_properties(parallelink PROPERTIES
    PREFIX ""                          # no "lib" prefix
    POSITION_INDEPENDENT_CODE ON
)

# -------------------------------------------------------------------
#  Holder daemon (keeps GPU memory alive across fio runs)
# -------------------------------------------------------------------
add_executable(plink-holder
    src/holder.cu
)

target_include_directories(plink-holder PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${LIBNVM_INCLUDE_DIRS}
)

target_link_libraries(plink-holder PRIVATE
    CUDA::cudart
)

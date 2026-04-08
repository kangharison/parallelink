cmake_minimum_required(VERSION 3.18)
project(fio-bam LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# -------------------------------------------------------------------
#  Paths
# -------------------------------------------------------------------
set(FIO_DIR    ${CMAKE_SOURCE_DIR}/extern/fio)
set(BAM_DIR    ${CMAKE_SOURCE_DIR}/extern/bam)

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
#  BAM headers
# -------------------------------------------------------------------
set(BAM_INCLUDE_DIRS
    ${BAM_DIR}/include
    ${BAM_DIR}/src
)

# -------------------------------------------------------------------
#  CUDA
# -------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# -------------------------------------------------------------------
#  Engine shared library: fio-engine-bam.so
#
#  fio loads external engines via dlopen("fio-engine-bam.so").
# -------------------------------------------------------------------
add_library(fio-engine-bam SHARED
    src/engine.c
    src/gpu_worker.cu
)

target_include_directories(fio-engine-bam PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${FIO_INCLUDE_DIRS}
    ${BAM_INCLUDE_DIRS}
)

target_link_libraries(fio-engine-bam PRIVATE
    CUDA::cudart
)

# fio expects the engine symbol to be visible
set_target_properties(fio-engine-bam PROPERTIES
    PREFIX ""                          # no "lib" prefix
    POSITION_INDEPENDENT_CODE ON
)

# -------------------------------------------------------------------
#  Holder daemon (keeps GPU memory alive across fio runs)
# -------------------------------------------------------------------
add_executable(bam-holder
    src/holder.cu
)

target_include_directories(bam-holder PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${BAM_INCLUDE_DIRS}
)

target_link_libraries(bam-holder PRIVATE
    CUDA::cudart
)

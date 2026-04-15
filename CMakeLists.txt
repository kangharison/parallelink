cmake_minimum_required(VERSION 3.18)
project(parallelink LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# -------------------------------------------------------------------
#  Build type: Debug / Release (default: Release)
#   Debug  : -g -O0 host + -G -g -O0 device (cuda-gdb breakpoints)
#   Release: -O2 host  + -lineinfo -O3 device
# -------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose build type: Debug Release" FORCE)
endif()
message(STATUS "parallelink build type: ${CMAKE_BUILD_TYPE}")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -g -O0")
else()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -O3")
endif()

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
    ${LIBNVM_DIR}/include/freestanding/include
    ${LIBNVM_DIR}/src
)

# -------------------------------------------------------------------
#  CUDA
# -------------------------------------------------------------------
find_package(CUDAToolkit REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# -------------------------------------------------------------------
#  libnvm (BaM) prebuilt shared library
#  Expected location: extern/bam/build/lib/libnvm.so
# -------------------------------------------------------------------
set(LIBNVM_LIB ${LIBNVM_DIR}/build/lib/libnvm.so)
if(NOT EXISTS ${LIBNVM_LIB})
    message(WARNING "libnvm.so not found at ${LIBNVM_LIB}. "
                    "Build extern/bam first (see README).")
endif()

# -------------------------------------------------------------------
#  Engine shared library: parallelink.so
#
#  fio loads external engines via dlopen("parallelink.so").
# -------------------------------------------------------------------
add_library(parallelink SHARED
    src/gpu_engine.c
    src/gpu_worker.cu
)

target_include_directories(parallelink PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${FIO_INCLUDE_DIRS}
    ${LIBNVM_INCLUDE_DIRS}
)

target_compile_definitions(parallelink PRIVATE
    _GNU_SOURCE
    PLINK_BUILD_TYPE="${CMAKE_BUILD_TYPE}"
)

# BaM headers use __CUDA__ to gate device-side code paths.
set_source_files_properties(src/gpu_worker.cu PROPERTIES
    COMPILE_DEFINITIONS "__CUDA__"
)

target_link_libraries(parallelink PRIVATE
    CUDA::cudart
    ${LIBNVM_LIB}
    pthread
)

# fio expects the engine symbol to be visible
set_target_properties(parallelink PROPERTIES
    PREFIX ""                          # no "lib" prefix
    POSITION_INDEPENDENT_CODE ON
)

# -------------------------------------------------------------------
#  bam-admin-cli: out-of-band NVMe admin command injector
#
#  Tiny standalone client that talks to the parallelink engine's
#  admin helper socket. No CUDA / no libnvm dependency — it only
#  speaks the wire protocol.
# -------------------------------------------------------------------
add_executable(bam-admin-cli
    src/bam_admin_cli.c
)

target_compile_definitions(bam-admin-cli PRIVATE _GNU_SOURCE)

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

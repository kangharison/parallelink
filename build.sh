#!/usr/bin/env bash
#
# parallelink end-to-end build script
#   1. fio         (extern/fio)
#   2. libnvm/BaM  (extern/bam)
#   3. parallelink (this repo)
#
# Target GPU: NVIDIA Blackwell (sm_100 / sm_120).
# Requires: gcc-11 / g++-11, CUDA 13.x, cmake >= 3.18.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIO_DIR="${ROOT}/extern/fio"
BAM_DIR="${ROOT}/extern/bam"
BAM_BUILD="${BAM_DIR}/build"
PLINK_BUILD="${ROOT}/build"

CUDA_ARCHS="${CUDA_ARCHS:-100;120}"     # Blackwell
JOBS="${JOBS:-$(nproc)}"

CC="${CC:-gcc-11}"
CXX="${CXX:-g++-11}"
CUDA_HOST_CXX="${CUDA_HOST_CXX:-g++-11}"
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"

echo "==> parallelink build"
echo "    CUDA_ARCHS = ${CUDA_ARCHS}"
echo "    JOBS       = ${JOBS}"
echo "    CC/CXX     = ${CC} / ${CXX}"
echo "    NVCC       = ${NVCC}"

# ------------------------------------------------------------------
# 0. Submodules
# ------------------------------------------------------------------
if [[ ! -f "${FIO_DIR}/Makefile" ]] || [[ ! -f "${BAM_DIR}/CMakeLists.txt" ]]; then
    echo "==> [0/3] git submodule update --init --recursive"
    git -C "${ROOT}" submodule update --init --recursive
fi

# ------------------------------------------------------------------
# 1. fio
# ------------------------------------------------------------------
echo "==> [1/3] Build fio"
pushd "${FIO_DIR}" >/dev/null
if [[ ! -f config-host.h ]]; then
    ./configure
fi
make -j"${JOBS}"
popd >/dev/null

# ------------------------------------------------------------------
# 2. libnvm / BaM
# ------------------------------------------------------------------
echo "==> [2/3] Build libnvm (BaM)"
mkdir -p "${BAM_BUILD}"
pushd "${BAM_BUILD}" >/dev/null
if [[ ! -f CMakeCache.txt ]]; then
    cmake .. \
        -Dno_smartio=true \
        -Dno_module=true \
        -Dno_fio=true \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DCMAKE_CXX_COMPILER="${CXX}" \
        -DCMAKE_C_COMPILER="${CC}" \
        -DCMAKE_CUDA_HOST_COMPILER="${CUDA_HOST_CXX}" \
        -DCMAKE_CUDA_COMPILER="${NVCC}" \
        -DCMAKE_CXX_FLAGS="-I${BAM_DIR}/include/freestanding/include" \
        -DCMAKE_C_FLAGS="-I${BAM_DIR}/include/freestanding/include"
fi
make -j"${JOBS}"
popd >/dev/null

if [[ ! -f "${BAM_BUILD}/lib/libnvm.so" ]]; then
    echo "ERROR: ${BAM_BUILD}/lib/libnvm.so was not produced." >&2
    exit 1
fi

# ------------------------------------------------------------------
# 3. parallelink
# ------------------------------------------------------------------
echo "==> [3/3] Build parallelink"
mkdir -p "${PLINK_BUILD}"
pushd "${PLINK_BUILD}" >/dev/null
if [[ ! -f CMakeCache.txt ]]; then
    cmake .. \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DCMAKE_CXX_COMPILER="${CXX}" \
        -DCMAKE_C_COMPILER="${CC}" \
        -DCMAKE_CUDA_HOST_COMPILER="${CUDA_HOST_CXX}" \
        -DCMAKE_CUDA_COMPILER="${NVCC}"
fi
make -j"${JOBS}"
popd >/dev/null

echo "==> Done."
echo "    ${PLINK_BUILD}/parallelink.so"
echo "    ${PLINK_BUILD}/plink-holder"
echo "    ${FIO_DIR}/fio"

#!/usr/bin/env bash
#
# parallelink end-to-end build script
#   1. fio             (extern/fio)
#   2. libnvm + module (extern/bam)
#   3. parallelink     (this repo)
#   4. collect into dist/
#
# Target GPU: NVIDIA Blackwell (sm_100 / sm_120).
# Requires: gcc-11 / g++-11, CUDA 13.x, cmake >= 3.18, kernel headers.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIO_DIR="${ROOT}/extern/fio"
BAM_DIR="${ROOT}/extern/bam"
BAM_BUILD="${BAM_DIR}/build"
PLINK_BUILD="${ROOT}/build"
DIST="${ROOT}/dist"

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
    echo "==> [0/4] git submodule update --init --recursive"
    git -C "${ROOT}" submodule update --init --recursive
fi

# ------------------------------------------------------------------
# 1. fio
# ------------------------------------------------------------------
echo "==> [1/4] Build fio"
pushd "${FIO_DIR}" >/dev/null
if [[ ! -f config-host.h ]]; then
    ./configure
fi
make -j"${JOBS}"
popd >/dev/null

# ------------------------------------------------------------------
# 2. libnvm (userspace lib + kernel module)
# ------------------------------------------------------------------
echo "==> [2/4] Build libnvm (BaM) userspace + kernel module"
mkdir -p "${BAM_BUILD}"
pushd "${BAM_BUILD}" >/dev/null
if [[ ! -f CMakeCache.txt ]]; then
    cmake .. \
        -Dno_smartio=true \
        -Dno_module=false \
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

# Build the kernel module. BaM generates build/module/Makefile from
# module/Makefile.in via cmake variable substitution.
if [[ -f module/Makefile ]]; then
    make -C module -j"${JOBS}"
else
    echo "WARN: ${BAM_BUILD}/module/Makefile not found — skipping .ko build" >&2
fi
popd >/dev/null

if [[ ! -f "${BAM_BUILD}/lib/libnvm.so" ]]; then
    echo "ERROR: ${BAM_BUILD}/lib/libnvm.so was not produced." >&2
    exit 1
fi

# ------------------------------------------------------------------
# 3. parallelink
# ------------------------------------------------------------------
echo "==> [3/4] Build parallelink"
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

# ------------------------------------------------------------------
# 4. Collect artifacts into dist/
# ------------------------------------------------------------------
echo "==> [4/4] Collect artifacts → ${DIST}"
rm -rf "${DIST}"
mkdir -p "${DIST}"

# fio binary
cp -f "${FIO_DIR}/fio" "${DIST}/fio"

# libnvm userspace shared library (runtime dep of parallelink.so)
cp -f "${BAM_BUILD}/lib/libnvm.so" "${DIST}/libnvm.so"

# libnvm kernel module (if built)
KO_PATH="$(find "${BAM_BUILD}/module" -maxdepth 2 -name 'libnvm.ko' 2>/dev/null | head -n1 || true)"
if [[ -n "${KO_PATH}" && -f "${KO_PATH}" ]]; then
    cp -f "${KO_PATH}" "${DIST}/libnvm.ko"
else
    echo "WARN: libnvm.ko not found — kernel headers may be missing" >&2
fi

# parallelink engine + holder daemon
cp -f "${PLINK_BUILD}/parallelink.so" "${DIST}/parallelink.so"
cp -f "${PLINK_BUILD}/plink-holder"   "${DIST}/plink-holder"

# Convenience runner: wraps fio with FIO_EXT_ENG_DIR + LD_LIBRARY_PATH
cat > "${DIST}/run-fio.sh" <<'RUN'
#!/usr/bin/env bash
# Launch the bundled fio with parallelink.so from this dist directory.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export FIO_EXT_ENG_DIR="${HERE}"
export LD_LIBRARY_PATH="${HERE}:${LD_LIBRARY_PATH:-}"
exec "${HERE}/fio" "$@"
RUN
chmod +x "${DIST}/run-fio.sh"

echo "==> Done. Artifacts in ${DIST}:"
ls -lh "${DIST}"

cat <<EOF

Next steps:
  sudo insmod ${DIST}/libnvm.ko                          # load kernel module
  echo 0000:xx:xx.x | sudo tee /sys/bus/pci/drivers/nvme/unbind
  echo 0000:xx:xx.x | sudo tee /sys/bus/pci/drivers/libnvm/bind
  ${DIST}/run-fio.sh --ioengine=external:parallelink.so ...
EOF

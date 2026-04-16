#!/usr/bin/env bash
#
# parallelink end-to-end build script
#   1. fio             (extern/fio)
#   2. libnvm + module (extern/bam)
#   3. parallelink     (this repo)
#   4. collect into dist/
#
# Target GPU: NVIDIA Blackwell (sm_100 / sm_120).
# Requires: gcc-12 / g++-12 (CUDA 13.x host compiler 지원 범위), CUDA 13.x,
#           cmake >= 3.18, kernel headers.
#
# 기본값은 gcc-12/g++-12. 다른 버전을 쓰려면 CC/CXX/CUDA_HOST_CXX env로 오버라이드.
# 예) CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 CUDA_HOST_CXX=/usr/bin/g++-13 ./build.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIO_DIR="${ROOT}/extern/fio"
BAM_DIR="${ROOT}/extern/bam"
BAM_BUILD="${BAM_DIR}/build"
LIBNVME_DIR="${ROOT}/extern/libnvme"
NVME_CLI_DIR="${ROOT}/extern/nvme-cli"
PLINK_BUILD="${ROOT}/build"
DIST="${ROOT}/dist"

CUDA_ARCHS="${CUDA_ARCHS:-100;120}"     # Blackwell
JOBS="${JOBS:-$(nproc)}"
BUILD_TYPE="${BUILD_TYPE:-Debug}"       # Debug | Release
NVCC="${NVCC:-/usr/local/cuda/bin/nvcc}"

# 호스트 컴파일러 자동 선택:
#   1) env로 넘어온 CC/CXX/CUDA_HOST_CXX가 있으면 그대로 사용 (최우선)
#   2) gcc-12/g++-12가 설치돼 있으면 그것을 사용 (CUDA 13.x 권장)
#   3) 아니면 시스템 기본 gcc/g++ 사용 — 단 주 버전이 12 또는 13일 때만 허용
# (CUDA 13.x host compiler 지원: GCC 8~14. 14 초과는 런타임에 실패 가능.)
_pick_compiler() {
    local preferred="$1" fallback="$2"
    if command -v "${preferred}" >/dev/null 2>&1; then
        echo "${preferred}"
    elif command -v "${fallback}" >/dev/null 2>&1; then
        echo "${fallback}"
    else
        echo ""
    fi
}
CC="${CC:-$(_pick_compiler gcc-12 gcc)}"
CXX="${CXX:-$(_pick_compiler g++-12 g++)}"
CUDA_HOST_CXX="${CUDA_HOST_CXX:-${CXX}}"

# 존재 검증 + 주 버전 확인 (nvcc의 모호한 에러를 선차단)
for _bin in "${CC}" "${CXX}" "${CUDA_HOST_CXX}"; do
    if [[ -z "${_bin}" ]] || ! command -v "${_bin}" >/dev/null 2>&1; then
        echo "ERROR: host compiler not found. 설치된 gcc/g++가 없습니다." >&2
        echo "       해결: sudo apt install gcc-12 g++-12" >&2
        echo "       또는 CC=... CXX=... CUDA_HOST_CXX=... 로 오버라이드" >&2
        exit 1
    fi
done
_CXX_MAJOR="$("${CXX}" -dumpversion 2>/dev/null | cut -d. -f1)"
if [[ "${_CXX_MAJOR}" -lt 8 || "${_CXX_MAJOR}" -gt 14 ]]; then
    echo "ERROR: CXX=${CXX} 버전 major=${_CXX_MAJOR} — CUDA 13.x는 GCC 8~14 지원." >&2
    exit 1
fi
echo "    selected CXX = ${CXX} (major=${_CXX_MAJOR})"

# CMake가 암묵적으로 읽는 CUDA 관련 env를 본 스크립트 범위에서 무력화.
# 사용자 쉘에 `export CUDAHOSTCXX=g++11` 같은 오타가 남아 있어도, 여기서 명시
# 선택한 CUDA_HOST_CXX가 항상 우선하도록 export 고정한다.
export CUDAHOSTCXX="${CUDA_HOST_CXX}"
export CUDACXX="${NVCC}"
unset CCBIN

echo "==> parallelink build"
echo "    BUILD_TYPE = ${BUILD_TYPE}"
echo "    CUDA_ARCHS = ${CUDA_ARCHS}"
echo "    JOBS       = ${JOBS}"
echo "    CC/CXX     = ${CC} / ${CXX}"
echo "    NVCC       = ${NVCC}"

# ------------------------------------------------------------------
# Apply BaM cmake patch (idempotent) — adds Debug/Release switching
# to extern/bam/CMakeLists.txt since upstream has it hardcoded.
# ------------------------------------------------------------------
for BAM_PATCH in "${ROOT}"/patches/bam-*.patch; do
    if [[ -f "${BAM_PATCH}" ]]; then
        if git -C "${BAM_DIR}" apply --reverse --check "${BAM_PATCH}" >/dev/null 2>&1; then
            echo "==> ${BAM_PATCH} already applied"
        else
            echo "==> Applying ${BAM_PATCH}"
            git -C "${BAM_DIR}" apply "${BAM_PATCH}"
        fi
    fi
done

# ------------------------------------------------------------------
# Apply libnvme patches (idempotent) — currently just the PLINK
# ioctl hook that redirects admin passthru through the parallelink
# admin socket.
# ------------------------------------------------------------------
for LIBNVME_PATCH in "${ROOT}"/patches/libnvme-*.patch; do
    if [[ -f "${LIBNVME_PATCH}" ]]; then
        if git -C "${LIBNVME_DIR}" apply --reverse --check "${LIBNVME_PATCH}" >/dev/null 2>&1; then
            echo "==> ${LIBNVME_PATCH} already applied"
        else
            echo "==> Applying ${LIBNVME_PATCH}"
            git -C "${LIBNVME_DIR}" apply "${LIBNVME_PATCH}"
        fi
    fi
done

# ------------------------------------------------------------------
# 0. Submodules
#
# libnvme / nvme-cli are checked out here but not yet built — the
# PLINK ioctl hook wiring (libnvme patch + meson build) lands in a
# later commit. For now we just make sure the trees exist so the
# upcoming build steps can assume them.
# ------------------------------------------------------------------
if [[ ! -f "${FIO_DIR}/Makefile" ]] \
   || [[ ! -f "${BAM_DIR}/CMakeLists.txt" ]] \
   || [[ ! -f "${LIBNVME_DIR}/meson.build" ]] \
   || [[ ! -f "${NVME_CLI_DIR}/meson.build" ]]; then
    echo "==> [0/6] git submodule update --init --recursive"
    git -C "${ROOT}" submodule update --init --recursive
fi

# ------------------------------------------------------------------
# 1. fio
# ------------------------------------------------------------------
echo "==> [1/6] Build fio"
pushd "${FIO_DIR}" >/dev/null
if [[ ! -f config-host.h ]]; then
    ./configure
fi
make -j"${JOBS}"
popd >/dev/null

# ------------------------------------------------------------------
# 2. libnvm (userspace lib + kernel module)
# ------------------------------------------------------------------
echo "==> [2/6] Build libnvm (BaM) userspace + kernel module"
mkdir -p "${BAM_BUILD}"
pushd "${BAM_BUILD}" >/dev/null
CACHED_TYPE="$(grep -E '^CMAKE_BUILD_TYPE' CMakeCache.txt 2>/dev/null | cut -d= -f2 || true)"
if [[ ! -f CMakeCache.txt || "${CACHED_TYPE}" != "${BUILD_TYPE}" ]]; then
    echo "    (re)configuring BaM (cached=${CACHED_TYPE:-none} requested=${BUILD_TYPE})"
    rm -f CMakeCache.txt
    cmake .. \
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
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
    if ! make -C module -j"${JOBS}"; then
        echo "WARN: libnvm.ko build failed (kernel headers mismatch?)" >&2
        echo "      Userspace parallelink build will continue." >&2
    fi
else
    echo "WARN: ${BAM_BUILD}/module/Makefile not found — skipping .ko build" >&2
fi
popd >/dev/null

if [[ ! -f "${BAM_BUILD}/lib/libnvm.so" ]]; then
    echo "ERROR: ${BAM_BUILD}/lib/libnvm.so was not produced." >&2
    exit 1
fi

# ------------------------------------------------------------------
# 3. parallelink (also produces libplink_hook.a for libnvme)
# ------------------------------------------------------------------
echo "==> [3/6] Build parallelink"
mkdir -p "${PLINK_BUILD}"
pushd "${PLINK_BUILD}" >/dev/null
CACHED_TYPE="$(grep -E '^CMAKE_BUILD_TYPE' CMakeCache.txt 2>/dev/null | cut -d= -f2 || true)"
if [[ ! -f CMakeCache.txt || "${CACHED_TYPE}" != "${BUILD_TYPE}" ]]; then
    echo "    (re)configuring parallelink (cached=${CACHED_TYPE:-none} requested=${BUILD_TYPE})"
    rm -f CMakeCache.txt
    cmake .. \
        -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
        -DCMAKE_CXX_COMPILER="${CXX}" \
        -DCMAKE_C_COMPILER="${CC}" \
        -DCMAKE_CUDA_HOST_COMPILER="${CUDA_HOST_CXX}" \
        -DCMAKE_CUDA_COMPILER="${NVCC}"
fi
make -j"${JOBS}"
popd >/dev/null

PLINK_HOOK_LIB="${PLINK_BUILD}/libplink_hook.a"
if [[ ! -f "${PLINK_HOOK_LIB}" ]]; then
    echo "ERROR: ${PLINK_HOOK_LIB} was not produced by parallelink build." >&2
    exit 1
fi

# ------------------------------------------------------------------
# 4. libnvme (with PLINK admin ioctl hook)
#
# Meson build. We inject:
#   -DPLINK           into c_args so src/nvme/ioctl.c compiles the
#                     call-site that goes through plink_ioctl_hook()
#   libplink_hook.a   into c_link_args so libnvme.so resolves the
#                     plink_ioctl_hook symbol at link time
# ------------------------------------------------------------------
echo "==> [4/6] Build libnvme with PLINK ioctl hook"
LIBNVME_BUILD="${LIBNVME_DIR}/build"
LIBNVME_BUILDTYPE="$(echo "${BUILD_TYPE}" | tr '[:upper:]' '[:lower:]')"

LIBNVME_MESON_ARGS=(
    -Dc_args="-DPLINK"
    -Dc_link_args="${PLINK_HOOK_LIB}"
    -Dbuildtype="${LIBNVME_BUILDTYPE}"
    -Dpython=disabled
    -Dopenssl=disabled
    -Dlibdbus=disabled
    -Dkeyutils=disabled
    -Djson-c=disabled
    -Ddocs=false
    -Dtests=false
)

if [[ ! -f "${LIBNVME_BUILD}/build.ninja" ]]; then
    meson setup "${LIBNVME_BUILD}" "${LIBNVME_DIR}" "${LIBNVME_MESON_ARGS[@]}"
else
    meson configure "${LIBNVME_BUILD}" "${LIBNVME_MESON_ARGS[@]}"
fi
ninja -C "${LIBNVME_BUILD}" -j"${JOBS}"

# ------------------------------------------------------------------
# 5. nvme-cli (linked against the PLINK-hooked libnvme)
#
# nvme-cli discovers libnvme via pkg-config. Point PKG_CONFIG_PATH
# at libnvme's meson uninstalled pkgconfig dir so it prefers our
# build over any system-installed libnvme. Also set RPATH so the
# bundled nvme binary finds libnvme.so from dist/ at runtime.
# ------------------------------------------------------------------
echo "==> [5/6] Build nvme-cli"
NVME_CLI_BUILD="${NVME_CLI_DIR}/build"

LIBNVME_PC_DIR="${LIBNVME_BUILD}/meson-uninstalled"
if [[ ! -d "${LIBNVME_PC_DIR}" ]]; then
    echo "ERROR: ${LIBNVME_PC_DIR} not found — libnvme build incomplete?" >&2
    exit 1
fi
export PKG_CONFIG_PATH="${LIBNVME_PC_DIR}:${PKG_CONFIG_PATH:-}"

#
# -Dlibnvme=disabled is critical: nvme-cli vendors libnvme under
# extern/nvme-cli/libnvme/ and would otherwise build that nested copy,
# ignoring the PLINK-hooked libnvme we just built at extern/libnvme.
# Disabling it forces nvme-cli to resolve libnvme via pkg-config —
# which PKG_CONFIG_PATH above points at our patched build's
# meson-uninstalled dir.
#
NVME_CLI_MESON_ARGS=(
    -Dbuildtype="${LIBNVME_BUILDTYPE}"
    -Dc_link_args="-Wl,-rpath,\$ORIGIN"
    -Dlibnvme=disabled
    -Dpython=disabled
    -Dopenssl=disabled
    -Dlibdbus=disabled
    -Dkeyutils=disabled
    -Djson-c=disabled
    -Ddocs=false
    -Dtests=false
)

if [[ ! -f "${NVME_CLI_BUILD}/build.ninja" ]]; then
    meson setup "${NVME_CLI_BUILD}" "${NVME_CLI_DIR}" "${NVME_CLI_MESON_ARGS[@]}"
else
    meson configure "${NVME_CLI_BUILD}" "${NVME_CLI_MESON_ARGS[@]}"
fi
ninja -C "${NVME_CLI_BUILD}" -j"${JOBS}"

# ------------------------------------------------------------------
# 6. Collect artifacts into dist/
# ------------------------------------------------------------------
echo "==> [6/6] Collect artifacts → ${DIST}"
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
cp -f "${PLINK_BUILD}/bam-admin-cli"  "${DIST}/bam-admin-cli"

# PLINK-hooked libnvme + nvme-cli
LIBNVME_SO="$(find "${LIBNVME_BUILD}" -maxdepth 2 -name 'libnvme.so*' -type f 2>/dev/null | head -n1 || true)"
if [[ -z "${LIBNVME_SO}" || ! -f "${LIBNVME_SO}" ]]; then
    echo "ERROR: libnvme.so not found under ${LIBNVME_BUILD}" >&2
    exit 1
fi
cp -f "${LIBNVME_SO}" "${DIST}/libnvme.so"

NVME_BIN="${NVME_CLI_BUILD}/nvme"
if [[ ! -x "${NVME_BIN}" ]]; then
    echo "ERROR: nvme-cli binary not found at ${NVME_BIN}" >&2
    exit 1
fi
cp -f "${NVME_BIN}" "${DIST}/nvme"

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

The bundled nvme-cli (${DIST}/nvme) links against a libnvme that
routes NVME_IOCTL_ADMIN_CMD through the parallelink admin socket
at ${PLINK_ADMIN_SOCK:-/tmp/nvme-admin.sock}. Run it while the fio
engine is up, e.g.:
  ${DIST}/nvme id-ctrl /dev/nvme0
EOF

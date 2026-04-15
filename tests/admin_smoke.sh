#!/usr/bin/env bash
# admin_smoke.sh: verify that bam-admin-cli id-ctrl returns a plausible
# Identify Controller response while a fio workload is running.
#
# Requires a working dist/ build (run ./build.sh first) and root-level
# access to /dev/libnvm0. Run from repo root.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIST="${ROOT}/dist"

if [[ ! -x "${DIST}/fio" || ! -x "${DIST}/bam-admin-cli" ]]; then
    echo "Missing dist/ artifacts. Run ./build.sh first." >&2
    exit 1
fi

export FIO_EXT_ENG_DIR="${DIST}"
export LD_LIBRARY_PATH="${DIST}:${LD_LIBRARY_PATH:-}"

TMP=$(mktemp -d)
trap 'rm -rf "${TMP}"; [[ -n "${FIO_PID:-}" ]] && kill "${FIO_PID}" 2>/dev/null || true' EXIT

# Background fio: long random read workload
"${DIST}/fio" \
    --thread=1 --numjobs=1 \
    --name=admin-smoke \
    --ioengine=external:parallelink.so \
    --rw=randread --bs=4k \
    --size=1G --runtime=30 --time_based \
    --gpu_warps=32 --n_queues=16 \
    > "${TMP}/fio.log" 2>&1 &
FIO_PID=$!

# Wait for admin socket to appear (up to 10s)
SOCK="/tmp/bam-admin-${FIO_PID}.sock"
for _ in $(seq 1 50); do
    [[ -S "${SOCK}" ]] && break
    sleep 0.2
done

if [[ ! -S "${SOCK}" ]]; then
    echo "FAIL: admin socket ${SOCK} never appeared" >&2
    echo "---- fio.log ----" >&2
    cat "${TMP}/fio.log" >&2
    exit 1
fi

# Issue Identify Controller
OUT="${TMP}/id-ctrl.txt"
"${DIST}/bam-admin-cli" --pid "${FIO_PID}" id-ctrl > "${OUT}" 2>&1

# Sanity: expect the response to be 4KB of hex dump and start with
# nonzero bytes (PCI vendor ID at offset 0 of Identify Controller).
if ! grep -q '^00000000  ' "${OUT}"; then
    echo "FAIL: no hex dump in response" >&2
    cat "${OUT}" >&2
    exit 1
fi

# First 2 bytes are VID; all-zero means the admin round-trip didn't
# actually land a real completion.
FIRST=$(awk '/^00000000  /{print $2$3; exit}' "${OUT}")
if [[ "${FIRST}" == "0000" || -z "${FIRST}" ]]; then
    echo "FAIL: Identify Controller returned all zeros (VID=${FIRST})" >&2
    cat "${OUT}" >&2
    exit 1
fi

echo "OK: Identify Controller VID=${FIRST}"
kill "${FIO_PID}" 2>/dev/null || true
wait "${FIO_PID}" 2>/dev/null || true
FIO_PID=

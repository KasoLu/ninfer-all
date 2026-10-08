#!/usr/bin/env bash
set -Eeuo pipefail
build=${NINFER_BUILD_DIR:-/workspace/ninfer-work/build}
# nvcc's concurrent temporary files can fill this rental's model volume before linking. Keep
# them in the host's tmpfs; the build's memory-derived worker limit still applies.
export TMPDIR
TMPDIR=$(mktemp -d /dev/shm/ninfer-build-XXXXXX)
trap 'rm -rf "$TMPDIR"' EXIT
ulimit -c 0
# The size-focused compressor occupied individual CPU cores for minutes in this build. Balance
# changes fatbinary packaging only; retain completed objects and use it for subsequent compiles.
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-} --compress-mode=balance"
echo 'nvcc fatbinary compression: balance'
export NINFER_BUILD_ID=$(/workspace/ninfer-work/py311/bin/python -c \
    'import json; d=json.load(open("/workspace/ninfer-work/source.json")); print(d["commit"][:12]+"-snapshot-"+d["source_sha256"][:12])')
if [ ! -f "$build/CMakeCache.txt" ] || [ ! -f "$build/build.ninja" ] || \
   [ ! -f "$build/src/runtime/engine/device_profiles_builtin.cpp" ]; then
    cmake -S . -B "$build" -G Ninja -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_CUDA_ARCHITECTURES=86 -DBUILD_TESTING=ON -DNINFER_BUILD_APPS=ON \
        -DNINFER_BUILD_BENCHMARKS=ON -DPython3_EXECUTABLE=/workspace/ninfer-work/py311/bin/python
fi
# Each nvcc can use several GiB; cgroup RAM, not the host's total, controls the bound.
jobs=$(/workspace/ninfer-work/py311/bin/python -c 'import os; from pathlib import Path; cap=Path("/sys/fs/cgroup/memory.max"); limit=cap.read_text().strip() if cap.exists() else "max"; ram=int(limit) if limit != "max" else os.sysconf("SC_PAGE_SIZE")*os.sysconf("SC_PHYS_PAGES"); print(max(1,min(os.cpu_count(),ram//(3*1024**3))))')
cmake --build "$build" -j "$jobs" "$@"
sha256sum "$build/apps/ninfer" "$build/apps/ninfer-serve" > ../jobs/build-binaries.sha256

#!/usr/bin/env bash
# Copies the image's binaries out of one NInfer build tree into <dist>/sm<arch>/ and records the
# Debian packages that provide their shared libraries, outside the CUDA libraries the runtime base
# image already carries, so the runtime stage installs exactly those. Used by the Dockerfile's
# build stage and by CI, whose build jobs hand the same layout to the image.
#
#   docker/stage-dist.sh <build dir> <arch> <dist dir>
set -euo pipefail
(( $# == 3 )) || { printf 'usage: stage-dist.sh <build dir> <arch> <dist dir>\n' >&2; exit 2; }
build_dir="$1" arch="$2" out="$3/sm$2"
binaries=(ninfer ninfer-serve ninfer-perplexity ninfer-calibrate)

mkdir -p -- "$out"
for binary in "${binaries[@]}"; do
  install -m 0755 -- "$build_dir/apps/$binary" "$out/$binary"
done
# ldd resolves through the build's own library path; dpkg -S names the owning package. CUDA's
# libraries come from the runtime base image at the same CUDA version, so they are left out.
ldd "${binaries[@]/#/$out/}" | awk '$2 == "=>" && $3 ~ /^\// { print $3 }' | sort -u \
  | while read -r library; do
      dpkg -S "$(readlink -f -- "$library")" 2>/dev/null || dpkg -S "$library" 2>/dev/null || true
    done \
  | cut -d: -f1 | sort -u \
  | grep -Ev '^(cuda-|libcublas|libcufft|libcurand|libcusolver|libcusparse|libnpp|libnvjpeg|libnccl|libcudnn|libc6$|libgcc-s1$|libstdc\+\+6$)' \
  > "$out/runtime-packages.txt" || true
printf 'staged %s: %s\n' "$out" "$(tr '\n' ' ' < "$out/runtime-packages.txt")"

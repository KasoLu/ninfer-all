# syntax=docker/dockerfile:1.7
#
# NInfer server image. ninfer, ninfer-serve, ninfer-perplexity and ninfer-calibrate are built once
# per architecture -- sm_86 (Ampere; the same cubins run on Ada sm_89) and sm_120a (consumer
# Blackwell), since a NInfer build targets exactly one -- next to the launcher and the downloader.
# The entrypoint runs the build that matches the GPU it is given (docker/entrypoint.sh).
#
#   docker build -t ninfer .                               both architectures, from source
#   docker build --build-arg ARCHS=86 -t ninfer .          one architecture
#   docker build --build-context dist=<dir> -t ninfer .    package prebuilt binaries (what CI does)
#
# A prebuilt dist directory has the layout docker/stage-dist.sh writes: sm86/ and/or sm120a/, each
# with the binaries and runtime-packages.txt.

ARG CUDA_VERSION=13.4.2
ARG UBUNTU_VERSION=26.04

# --- toolchain: also the CI build environment ------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS toolchain
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ccache \
        cmake \
        file \
        git \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libcurl4-openssl-dev \
        libswscale-dev \
        ninja-build \
        pkg-config \
        python3 \
        python3-jinja2 \
        rsync \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# --- build -----------------------------------------------------------------------------------------
FROM toolchain AS build
ARG ARCHS="86 120a"
ARG JOBS=""
WORKDIR /src
COPY . .

# Keep source mtimes stable only when contents match; restored/older checkouts must still rebuild
# changed inputs. Packages, the Dockerfile, the build script, CMake scripts and the architecture
# identify the configuration: removed flags and changed defaults must not reuse CMakeCache.txt.
# Lock the mutable Ninja trees and copy deliverables out of the transient cache mount.
RUN --mount=type=cache,id=ninfer-build,target=/build,sharing=locked \
    --mount=type=cache,target=/ccache \
    export CCACHE_DIR=/ccache CCACHE_MAXSIZE=20G NINFER_TESTS=OFF \
    && if [ -n "$JOBS" ]; then export NINFER_JOBS="$JOBS"; fi \
    && find . -type f \( -path ./Dockerfile -o -path ./scripts/build-native.sh \
        -o -name CMakeLists.txt -o -name '*.cmake' \) -exec sha256sum {} + > /build/configuration \
    && dpkg-query -W >> /build/configuration \
    && LC_ALL=C sort -o /build/configuration /build/configuration \
    && rsync --recursive --links --checksum --delete /src/ /build/src/ \
    && for arch in $ARCHS; do \
         build_dir="/build/$( { cat /build/configuration; echo "arch $arch"; } | sha256sum | cut -d ' ' -f 1)" \
         && NINFER_CUDA_ARCH="$arch" NINFER_BUILD_DIR="$build_dir" /build/src/scripts/build-native.sh configure \
         && NINFER_BUILD_DIR="$build_dir" /build/src/scripts/build-native.sh target \
              ninfer ninfer-serve ninfer-perplexity ninfer-calibrate \
         && /build/src/docker/stage-dist.sh "$build_dir" "$arch" /dist || exit 1; \
       done \
    && ccache --show-stats

FROM scratch AS dist
COPY --from=build /dist/ /

# --- runtime ---------------------------------------------------------------------------------------
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime
ARG DEBIAN_FRONTEND=noninteractive
COPY --from=dist / /opt/ninfer/
# chmod: a dist that went through a CI artifact has lost its executable bits.
RUN chmod 0755 /opt/ninfer/sm*/ninfer* \
    && apt-get update \
    && sort -u /opt/ninfer/sm*/runtime-packages.txt \
       | xargs apt-get install --yes --no-install-recommends aria2 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* /opt/ninfer/sm*/runtime-packages.txt

# The CUDA runtime image ships forward-compatibility libraries in /usr/local/cuda*/compat (a newer
# libcuda.so than the host driver). Forward compatibility is supported only on datacenter GPUs; on
# any GeForce card the loader picks these up and every CUDA call fails at startup with
#   cudaErrorCompatNotSupportedOnDevice: forward compatibility was attempted on non supported HW
# Removing them lets the container use the host driver through ordinary CUDA minor-version
# compatibility: any driver of the CUDA 13 branch (580 or newer) runs these binaries.
RUN rm -rf /usr/local/cuda*/compat

COPY scripts/run.sh scripts/download-model.sh /opt/ninfer/
COPY docker/entrypoint.sh /usr/local/bin/ninfer-entrypoint

# nvidia-smi (the "utility" capability) is how the entrypoint reads the GPU's compute capability.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    NINFER_MODEL_DIR=/models \
    NINFER_DEVICE_PROFILES=/cache/device-profiles.json \
    NINFER_HOST=0.0.0.0 \
    NINFER_PORT=8080
VOLUME ["/models", "/grafts", "/cache"]
WORKDIR /models
EXPOSE 8080
STOPSIGNAL SIGTERM
ENTRYPOINT ["ninfer-entrypoint"]
CMD ["run", "qwen38-27b"]

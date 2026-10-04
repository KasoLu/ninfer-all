#!/usr/bin/env bash
# Makes the Dockerfile's toolchain stage available locally as ninfer-toolchain:ci and prints its
# registry reference. The stage is published as ghcr.io/<owner>/<repo>-toolchain under a tag that
# hashes its definition (the global ARGs and the stage), so it is built once per change: pulled when
# the registry has that tag, built here otherwise, and then pushed with PUSH=1 for later runs.
#
#   PUSH=0|1 .github/scripts/toolchain-image.sh      (after docker login to ghcr.io)
set -euo pipefail
ref="ghcr.io/${GITHUB_REPOSITORY,,}-toolchain:$(sed -n '/^ARG CUDA_VERSION/,/^FROM toolchain AS build/p' \
  Dockerfile | sha256sum | cut -c1-16)"

if docker pull --quiet "$ref" > /dev/null 2>&1; then
  printf 'pulled %s\n' "$ref" >&2
else
  printf 'building %s\n' "$ref" >&2
  docker build --target toolchain --tag "$ref" \
    --label "org.opencontainers.image.source=$GITHUB_SERVER_URL/$GITHUB_REPOSITORY" . >&2
  if [[ "${PUSH:-0}" == 1 ]]; then
    docker push --quiet "$ref" >&2 || printf 'could not push %s; using the local build\n' "$ref" >&2
  fi
fi
docker tag "$ref" ninfer-toolchain:ci
printf '%s\n' "$ref"

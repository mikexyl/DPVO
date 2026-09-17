#!/usr/bin/env bash
# All package installation and native compilation happen inside Docker.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
native_image="${DPVO_JETSON_NATIVE_IMAGE:-dpvo:jetson-jp62-native}"
docker build -f "$root/deploy/jetson/Dockerfile.jp62" \
    --build-arg "DPVO_BASE_IMAGE=${DPVO_BASE_IMAGE:-nvcr.io/nvidia/pytorch:24.12-py3-igpu}" \
    --build-arg "MAX_JOBS=${DPVO_BUILD_JOBS:-2}" \
    -t "$native_image" "$root"
exec docker build -f "$root/deploy/jetson/Dockerfile.jp62.runtime" \
    --build-arg "DPVO_NATIVE_IMAGE=$native_image" \
    -t "${DPVO_JETSON_IMAGE:-dpvo:jetson-jp62-viser}" "$root"

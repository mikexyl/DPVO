#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
dockerfile="$root/deploy/jetson/Dockerfile"
if [[ "${DPVO_OFFLINE:-0}" == 1 ]]; then
    dockerfile="$dockerfile.offline"
fi
exec docker build -f "$dockerfile" \
    --build-arg "DPVO_BASE_IMAGE=${DPVO_BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.05-py3}" \
    --build-arg "MAX_JOBS=${DPVO_BUILD_JOBS:-2}" \
    -t "${DPVO_JETSON_IMAGE:-dpvo:jetson-jp7}" "$root"

#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
image="${DPVO_JETSON_IMAGE:-dpvo:jetson-jp7}"
data="${DPVO_DATA_DIR:-$root/data}"
weights="${DPVO_WEIGHTS:-$root/dpvo.pth}"
output="${DPVO_OUTPUT_DIR:-$root/results/jetson}"
test -f "$weights"
test -d "$data"
mkdir -p "$output"
exec docker run --rm --runtime=nvidia --gpus all --network=host \
    --shm-size=1g --user "$(id -u):$(id -g)" -e HOME=/tmp \
    -v "$data:/data:ro" -v "$weights:/models/dpvo.pth:ro" \
    -v "$output:/output" "$image" "$@"

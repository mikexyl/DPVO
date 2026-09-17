#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
robot="${1:?Usage: prepare_robot.sh robot0}"
[[ "$robot" =~ ^[a-zA-Z][a-zA-Z0-9_]*$ ]]
fleet="$(realpath "${DPVO_FLEET_DIR:-$here/generated}")"
models="$(realpath "${DPVO_MODEL_DIR:-$here/models}")"
output="${DPVO_OUTPUT_DIR:-$here/output/$robot}"
image="${DPVO_JETSON_IMAGE:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["image"])' "$fleet/$robot.json")}"
test -s "$models/dpvo.pth"
test -s "$models/ORBvoc.txt"
mkdir -p "$output"
docker run --rm --network host --runtime=nvidia -e HF_HUB_OFFLINE=0 \
    -e "HF_HUB_DISABLE_XET=${DPVO_HF_HUB_DISABLE_XET:-1}" \
    -v "$models:/models" "$image" python deploy/jetson/multi_robot/prepare_models.py
docker run --rm --network host --runtime=nvidia -v "$models:/models:ro" -v "$(realpath "$output"):/output" \
    "$image" python deploy/jetson/tensorrt_encoder.py --network /models/dpvo.pth \
    --height 240 --width 384 --output /output/engines-240x384

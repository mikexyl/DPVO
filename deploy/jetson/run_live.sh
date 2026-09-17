#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
weights="${DPVO_WEIGHTS:-$root/dpvo.pth}"
output="${DPVO_OUTPUT_DIR:-$root/results/jetson}"
web_port="${DPVO_WEB_PORT:-9090}"
grpc_port="${DPVO_GRPC_PORT:-9876}"
test -f "$weights"
mkdir -p "$output/live"
devices=(--device /dev/bus/usb)
for device in /dev/video*; do
    [[ -e "$device" ]] && devices+=(--device "$device")
done
extra=()
mode=(-d)
if [[ "${DPVO_FOREGROUND:-0}" == 1 ]]; then
    mode=(--rm)
fi
if [[ "${DPVO_START_PAUSED:-0}" == 1 ]]; then
    extra+=(--start-paused)
fi
if [[ -n "${DPVO_WEB_ORIGIN:-}" ]]; then
    extra+=(--web-origin "$DPVO_WEB_ORIGIN")
fi
exec docker run "${mode[@]}" --name "${DPVO_CONTAINER_NAME:-dpvo-realsense}" \
    --runtime=nvidia --gpus all --network=host --shm-size=1g \
    --log-opt max-size=10m --log-opt max-file=2 \
    "${devices[@]}" -v /run/udev:/run/udev:ro \
    -v "$weights:/models/dpvo.pth:ro" -v "$output:/output" \
    "${DPVO_LIVE_IMAGE:-dpvo:jetson-live}" \
    python deploy/jetson/live_realsense.py \
    --viewer "${DPVO_VIEWER:-rerun}" \
    --web-port "$web_port" --grpc-port "$grpc_port" \
    --viewer-fps "${DPVO_VIEWER_FPS:-5}" \
    --camera-fps "${DPVO_CAMERA_FPS:-30}" \
    --stride "${DPVO_CAMERA_STRIDE:-3}" "${extra[@]}" "$@"

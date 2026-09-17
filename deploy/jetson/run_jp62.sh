#!/usr/bin/env bash
# Isolated launch: no system service, clock, network, or other sensor changes.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
weights="${DPVO_WEIGHTS:-$root/dpvo.pth}"
output="${DPVO_OUTPUT_DIR:-$root/results/jetpack62}"
image="${DPVO_JETSON_IMAGE:-dpvo:jetson-jp62-viser}"
name="${DPVO_CONTAINER_NAME:-dpvo-jp62-viewer}"
test -f "$weights"
mkdir -p "$output/live"
# Grant access only to the selected D455F, not the board's other USB sensors.
candidates=()
for sysdevice in /sys/bus/usb/devices/*; do
    [[ -f "$sysdevice/idVendor" && -f "$sysdevice/idProduct" ]] || continue
    [[ "$(<"$sysdevice/idVendor")" == 8086 && "$(<"$sysdevice/idProduct")" == 0b5c ]] || continue
    if [[ -n "${DPVO_CAMERA_SERIAL:-}" && -f "$sysdevice/serial" ]]; then
        [[ "$(<"$sysdevice/serial")" == "$DPVO_CAMERA_SERIAL" ]] || continue
    fi
    candidates+=("$sysdevice")
done
if [[ ${#candidates[@]} != 1 ]]; then
    echo 'Connect one D455F or select it with DPVO_CAMERA_SERIAL.' >&2
    exit 1
fi
camera="${candidates[0]}"
printf -v usb_device '/dev/bus/usb/%03d/%03d' "$(<"$camera/busnum")" "$(<"$camera/devnum")"
# The SDK wheel uses the kernel video backend. Map only class devices whose
# sysfs ancestry belongs to this camera, excluding the carrier's CSI cameras.
devices=(--device "$usb_device")
camera_path="$(readlink -f "$camera")"
for class_device in /sys/class/video4linux/* /sys/class/hidraw/*; do
    [[ -e "$class_device/device" ]] || continue
    device_path="$(readlink -f "$class_device/device")"
    if [[ "$device_path" == "$camera_path/"* ]]; then
        devices+=(--device "/dev/${class_device##*/}")
    fi
done
extra=()
if [[ -n "${DPVO_CAMERA_SERIAL:-}" ]]; then
    extra+=(--serial "$DPVO_CAMERA_SERIAL")
fi
exec docker run -d --name "$name" --restart unless-stopped \
    --runtime=nvidia --network=bridge -p "${DPVO_WEB_PORT:-9090}:9090" \
    --shm-size=1g --log-opt max-size=10m --log-opt max-file=2 \
    "${devices[@]}" -v /run/udev:/run/udev:ro \
    -v "$weights:/models/dpvo.pth:ro" -v "$output:/output" \
    "$image" python deploy/jetson/live_realsense.py \
    --viewer viser --web-port 9090 --start-paused "${extra[@]}" \
    --camera-format "${DPVO_CAMERA_FORMAT:-yuyv}" \
    --camera-fps "${DPVO_CAMERA_FPS:-8}" --stride "${DPVO_CAMERA_STRIDE:-1}" \
    --trt-encoders /output/engines-240x384 "$@"

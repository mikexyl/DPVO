#!/usr/bin/env bash
# Maps only the selected camera. Never modifies host services or other containers.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
robot="${1:?Usage: run_robot.sh robot0}"
[[ "$robot" =~ ^[a-zA-Z][a-zA-Z0-9_]*$ ]]
fleet="$(realpath "${DPVO_FLEET_DIR:-$here/generated}")"
models="$(realpath "${DPVO_MODEL_DIR:-$here/models}")"
output="${DPVO_OUTPUT_DIR:-$here/output/$robot}"
read_json() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"; }
image="${DPVO_JETSON_IMAGE:-$(read_json "$fleet/$robot.json" image)}"
serial="$(read_json "$fleet/$robot.json" camera_serial)"
domain="$(read_json "$fleet/fleet.json" ros_domain_id)"
test -s "$models/dpvo.pth"
test -s "$models/ORBvoc.txt"
test -s "$models/loop_frontend/ready.json"
test -s "$output/engines-240x384/manifest.json"
camera_type="$(read_json "$fleet/$robot.json" camera_type)"
if [[ "$camera_type" == zed ]]; then
    test -s "$models/SN$serial.conf"
    video="$(python3 "$here/select_zed.py" --serial "$serial")"
    devices=(--device "$video:/dev/dpvo_camera")
else
    candidates=()
    for usb in /sys/bus/usb/devices/*; do
        [[ -f "$usb/idVendor" && -f "$usb/idProduct" ]] || continue
        [[ "$(<"$usb/idVendor")" == 8086 && "$(<"$usb/idProduct")" == 0b5c ]] || continue
        if [[ -n "$serial" ]]; then
            [[ -f "$usb/serial" && "$(<"$usb/serial")" == "$serial" ]] || continue
        fi
        candidates+=("$usb")
    done
    if [[ ${#candidates[@]} != 1 ]]; then
        echo 'Connect exactly one D455F, or set camera_serial in fleet.yaml and regenerate.' >&2
        exit 1
    fi
    camera="${candidates[0]}"
    printf -v bus '/dev/bus/usb/%03d/%03d' "$(<"$camera/busnum")" "$(<"$camera/devnum")"
    devices=(--device "$bus")
    camera_path="$(readlink -f "$camera")"
    for entry in /sys/class/video4linux/* /sys/class/hidraw/*; do
        [[ -e "$entry/device" ]] || continue
        device_path="$(readlink -f "$entry/device")"
        [[ "$device_path" == "$camera_path/"* ]] && devices+=(--device "/dev/${entry##*/}")
    done
fi
exec docker run -d --init --name "${DPVO_CONTAINER_NAME:-dpvo-online-$robot}" \
    --restart "${DPVO_CONTROLLER_RESTART_POLICY:-always}" --runtime=nvidia --network host --shm-size=1g \
    --log-opt max-size=10m --log-opt max-file=2 "${devices[@]}" \
    -e "ROS_DOMAIN_ID=$domain" -e CYCLONEDDS_URI=file:///fleet/cyclonedds.xml \
    -v /run/udev:/run/udev:ro -v "$fleet:/fleet:ro" -v "$models:/models:ro" \
    -v "$(realpath "$output"):/output" "$image" \
    python -m dpvo_multi_robot.online_control --ros-args \
    --params-file "/fleet/$robot.control.yaml" -r "__ns:=/$robot"

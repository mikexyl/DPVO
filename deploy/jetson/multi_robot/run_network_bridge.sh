#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
robot="${1:?Usage: run_network_bridge.sh robot0}"
[[ "$robot" =~ ^[a-zA-Z][a-zA-Z0-9_]*$ ]]
fleet="$(realpath "${DPVO_FLEET_DIR:-$here/generated}")"
transport="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("network", {}).get("transport", "zenoh"))' "$fleet/fleet.json")"
if [[ "$transport" != zenoh ]]; then
    echo "Bridge disabled: fleet transport is $transport" >&2
    exit 1
fi
test -s "$fleet/$robot.bridge.json"
exec docker run -d --init --name "${DPVO_BRIDGE_CONTAINER_NAME:-dpvo-network-$robot}" \
    --restart always --network host --log-opt max-size=10m --log-opt max-file=2 \
    -e CYCLONEDDS_URI=file:///fleet/cyclonedds.xml -e ROS_DISTRO=jazzy \
    -e RUST_LOG=warn -v "$fleet:/fleet:ro" \
    "${DPVO_BRIDGE_IMAGE:-dpvo:network-bridge-1.10.1}" -c "/fleet/$robot.bridge.json"

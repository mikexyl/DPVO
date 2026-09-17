#!/usr/bin/env bash
# A UI-only ROS client for the existing robot controller. No camera/GPU access.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
robot="${1:?Usage: run_robot_panel.sh robot0}"
[[ "$robot" =~ ^[a-zA-Z][a-zA-Z0-9_]*$ ]]
fleet="$(realpath "${DPVO_FLEET_DIR:-$here/generated}")"
read_json() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"; }
image="${DPVO_JETSON_IMAGE:-$(read_json "$fleet/$robot.json" image)}"
domain="$(read_json "$fleet/fleet.json" ros_domain_id)"
exec docker run -d --init --name "${DPVO_PANEL_CONTAINER_NAME:-dpvo-local-panel-$robot}" \
    --restart unless-stopped --network host \
    --log-opt max-size=10m --log-opt max-file=2 \
    -e "ROS_DOMAIN_ID=$domain" -e CYCLONEDDS_URI=file:///fleet/cyclonedds.xml \
    -v "$fleet:/fleet:ro" "$image" \
    python -m dpvo_multi_robot.online_viewer --ros-args \
    -r "__node:=dpvo_local_panel_$robot" \
    -p "robot_ids:=[$robot]" -p "web_port:=${DPVO_LOCAL_WEB_PORT:-9090}"

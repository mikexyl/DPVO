#!/usr/bin/env bash
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fleet="$(realpath "${DPVO_FLEET_DIR:-$here/generated}")"
output="${DPVO_OUTPUT_DIR:-$here/output/coordinator}"
test -f "$fleet/coordinator.yaml"
domain="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["ros_domain_id"])' "$fleet/fleet.json")"
mkdir -p "$output"
image="${DPVO_COORDINATOR_IMAGE:-$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["coordinator"].get("image", "dpvo:online-coordinator"))' "$fleet/fleet.json")}"
exec docker run -d --init --name "${DPVO_CONTAINER_NAME:-dpvo-online-coordinator}" \
    --restart unless-stopped --network host --log-opt max-size=10m --log-opt max-file=2 \
    -e "ROS_DOMAIN_ID=$domain" -e CYCLONEDDS_URI=file:///fleet/cyclonedds.xml \
    -v "$fleet:/fleet:ro" -v "$(realpath "$output"):/output" \
    "$image" \
    python deploy/jetson/multi_robot/coordinator.py

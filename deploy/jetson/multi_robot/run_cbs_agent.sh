#!/usr/bin/env bash
# Idle ROS control process; one native local optimizer only while an epoch runs.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
robot="${1:?Usage: run_cbs_agent.sh robot0}"
[[ "$robot" =~ ^[a-zA-Z][a-zA-Z0-9_]*$ ]]
fleet="$(realpath "${DPVO_FLEET_DIR:-$here/generated}")"
read_json() { python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$1" "$2"; }
domain="$(read_json "$fleet/fleet.json" ros_domain_id)"
profile="$(read_json "$fleet/$robot.json" profile)"
case "$profile" in
  jetpack72) default_image=dpvo:cbs-agent-jp72 ;;
  jetpack62) default_image=dpvo:cbs-agent-jp62 ;;
esac
output="${DPVO_CBS_OUTPUT_DIR:-$here/output/$robot/cbs-agent}"
mkdir -p "$output"
exec docker run -d --init --name "${DPVO_CBS_CONTAINER_NAME:-dpvo-cbs-agent-$robot}" \
    --restart unless-stopped --network host --cpus "${DPVO_CBS_CPUS:-2}" \
    --log-opt max-size=10m --log-opt max-file=2 \
    -e "ROS_DOMAIN_ID=$domain" -e CYCLONEDDS_URI=file:///fleet/cyclonedds.xml \
    -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=1 \
    -v "$fleet:/fleet:ro" -v "$(realpath "$output"):/output" \
    "${DPVO_CBS_AGENT_IMAGE:-$default_image}" \
    python -m dpvo_multi_robot.cbs_agent --ros-args -r "__ns:=/$robot" -p "robot_id:=$robot"

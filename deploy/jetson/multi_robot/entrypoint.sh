#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/jazzy/setup.bash
source /opt/online_ws/install/setup.bash
# ROS/OpenCV add system UCX; NVIDIA Torch needs its matching bundled UCX.
if [[ -d /opt/hpcx/ucx/lib ]]; then
    export LD_LIBRARY_PATH="/opt/hpcx/ucx/lib:${LD_LIBRARY_PATH:-}"
fi
export PYTHONPATH="/opt/dpvo:${PYTHONPATH:-}"
exec "$@"

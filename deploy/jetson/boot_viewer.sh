#!/usr/bin/env bash
# Installed as /usr/local/bin/dpvo-viewer-launch by the service setup.
set -euo pipefail
snapshot=/run/dpvo-viewer.clocks
if [[ "${1:-}" == --restore-clocks ]]; then
    if [[ -f "$snapshot" ]]; then
        jetson_clocks --restore "$snapshot"
    fi
    exit 0
fi
: "${DPVO_ROOT:?Set DPVO_ROOT in /etc/dpvo-viewer.env}"
if [[ "${DPVO_PERFORMANCE_CLOCKS:-0}" == 1 ]]; then
    if [[ ! -f "$snapshot" ]]; then
        jetson_clocks --store "$snapshot"
    fi
    python3 "$DPVO_ROOT/deploy/jetson/viewer_clocks.py" \
        "${DPVO_OUTPUT_DIR:-$DPVO_ROOT/results/jetson}/live/status.json" "$snapshot" &
fi
export DPVO_VIEWER=viser DPVO_FOREGROUND=1 DPVO_START_PAUSED=1
exec bash "$DPVO_ROOT/deploy/jetson/run_live.sh"

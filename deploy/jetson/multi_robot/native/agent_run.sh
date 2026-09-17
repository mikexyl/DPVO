#!/usr/bin/env bash
set -e
export LD_LIBRARY_PATH="/opt/cbs/lib"
exec /opt/cbs/bin/cbs_agent "$@"

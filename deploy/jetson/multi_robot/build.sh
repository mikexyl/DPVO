#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$root"
profile="${1:?Usage: build.sh jetpack62|jetpack72|coordinator}"
if [[ "$profile" == coordinator ]]; then
    test -f deploy/jetson/multi_robot/bundle/manifest.json
    exec docker build --network host -f deploy/jetson/multi_robot/Dockerfile.coordinator \
        -t "${DPVO_COORDINATOR_IMAGE:-dpvo:online-coordinator}" .
fi
case "$profile" in
    jetpack62) base=dpvo:jetson-jp62-viser; tag=dpvo:online-jp62; numpy=1 ;;
    jetpack72) base=dpvo:jetson-viser; tag=dpvo:online-jp72; numpy=2 ;;
    *) echo 'Choose jetpack62, jetpack72, or coordinator' >&2; exit 1 ;;
esac
exec docker build --network host -f deploy/jetson/multi_robot/Dockerfile.robot \
    --build-arg "DPVO_BASE_IMAGE=${DPVO_BASE_IMAGE:-$base}" \
    --build-arg "DPVO_NUMPY_MAJOR=$numpy" --build-arg "DPVO_BUILD_JOBS=${DPVO_BUILD_JOBS:-2}" \
    -t "${DPVO_JETSON_IMAGE:-$tag}" .

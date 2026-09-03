#!/usr/bin/env bash
set -eo pipefail

DPVO_DEPLOY_ROOT="${DPVO_ROOT:-/home/mikexyl/workspaces/dpvo_ws/src/DPVO}"
CBS_DEPENDENCY_PREFIX="${DPVO_CBS_DEPENDENCY_PREFIX:-/home/mikexyl/workspaces/sb_slam_ros2/install}"
CBS_BUILD_DIR="${DPVO_CBS_BUILD_DIR:-$DPVO_DEPLOY_ROOT/build/cbs}"
CBS_RUNTIME_PATH="$CBS_DEPENDENCY_PREFIX/gtsam/lib;$CBS_DEPENDENCY_PREFIX/aria_common/lib;$CBS_DEPENDENCY_PREFIX/aria_viz/lib"
export LD_LIBRARY_PATH="$CBS_DEPENDENCY_PREFIX/gtsam/lib:$CBS_DEPENDENCY_PREFIX/aria_common/lib:$CBS_DEPENDENCY_PREFIX/aria_viz/lib:${LD_LIBRARY_PATH:-}"

cmake \
  -S "$DPVO_DEPLOY_ROOT/cbs" \
  -B "$CBS_BUILD_DIR" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="$CBS_DEPENDENCY_PREFIX" \
  -DCMAKE_BUILD_RPATH="$CBS_RUNTIME_PATH" \
  -DCBS_BUILD_LEGACY_EXAMPLES=OFF \
  -DBUILD_TESTING=ON
cmake --build "$CBS_BUILD_DIR" --parallel "${DPVO_CBS_BUILD_JOBS:-8}"
ctest --test-dir "$CBS_BUILD_DIR" --output-on-failure

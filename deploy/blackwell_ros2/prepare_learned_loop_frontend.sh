#!/usr/bin/env bash
set -euo pipefail

DPVO_DEPLOY_ROOT="${DPVO_ROOT:-/home/mikexyl/workspaces/dpvo_ws/src/DPVO}"
DPVO_MODEL_ROOT="${DPVO_LEARNED_MODEL_ROOT:-/data3/mikexyl/models/dpvo_loop_frontend}"
DPVO_MEGALOC_REPO="$DPVO_MODEL_ROOT/MegaLoc"
DPVO_XFEAT_REPO="$DPVO_MODEL_ROOT/accelerated_features"
MEGALOC_COMMIT="5fe0dd697c4a70ba3e23607f6716ab3c606b16db"
XFEAT_COMMIT="e92685f57f8318b18725c5c8c0bd28c7fe188d9a"

mkdir -p "$DPVO_MODEL_ROOT"

if [[ ! -d "$DPVO_MEGALOC_REPO/.git" ]]; then
  git clone https://github.com/gmberton/MegaLoc.git "$DPVO_MEGALOC_REPO"
  git -C "$DPVO_MEGALOC_REPO" checkout --detach "$MEGALOC_COMMIT"
fi

if [[ ! -d "$DPVO_XFEAT_REPO/.git" ]]; then
  git clone https://github.com/verlab/accelerated_features.git "$DPVO_XFEAT_REPO"
  git -C "$DPVO_XFEAT_REPO" checkout --detach "$XFEAT_COMMIT"
fi

if [[ "$(git -C "$DPVO_MEGALOC_REPO" rev-parse HEAD)" != "$MEGALOC_COMMIT" ]]; then
  echo "MegaLoc cache is not at the validated commit: $DPVO_MEGALOC_REPO" >&2
  exit 1
fi
if [[ "$(git -C "$DPVO_XFEAT_REPO" rev-parse HEAD)" != "$XFEAT_COMMIT" ]]; then
  echo "XFeat cache is not at the validated commit: $DPVO_XFEAT_REPO" >&2
  exit 1
fi

export TORCH_HOME="${TORCH_HOME:-$DPVO_MODEL_ROOT/torch}"
export HF_HOME="${HF_HOME:-$DPVO_MODEL_ROOT/huggingface}"
export DPVO_MEGALOC_REPO
export DPVO_XFEAT_REPO
export PATH="/home/mikexyl/.pixi/bin:$PATH"

cd "$DPVO_DEPLOY_ROOT"
pixi run --manifest-path "$DPVO_DEPLOY_ROOT/deploy/blackwell_ros2/pixi.toml" \
  python - <<'PY'
import os

import torch

from dpvo.loop_closure.learned_frontend import (
    MegaLocDescriptorExtractor,
    XFeatFrontend,
)

megaloc = MegaLocDescriptorExtractor(os.environ["DPVO_MEGALOC_REPO"])
global_descriptor = megaloc(
    torch.zeros(1, 3, 224, 320, dtype=torch.uint8, device="cuda")
)
assert global_descriptor.shape == (1, 8448)

xfeat = XFeatFrontend(os.environ["DPVO_XFEAT_REPO"], top_k=256)
images = torch.rand(2, 3, 128, 160, device="cuda")
features = xfeat.detect(images)
matches = xfeat.matcher({"image0": features[0], "image1": features[1]})
assert matches["matches"].shape[-1] == 2
print("MegaLoc + XFeat/LighterGlue model cache is ready")
PY

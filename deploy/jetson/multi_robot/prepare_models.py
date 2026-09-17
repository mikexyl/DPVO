"""Run once in each robot image with a writable /models mount and network access."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import torch
from dpvo.loop_closure.learned_frontend import MegaLocDescriptorExtractor, XFeatFrontend


def main():
    root = Path('/models/loop_frontend')
    root.mkdir(parents=True, exist_ok=True)
    versions = {
        'MegaLoc': ('https://github.com/gmberton/MegaLoc.git', '5fe0dd697c4a70ba3e23607f6716ab3c606b16db'),
        'accelerated_features': ('https://github.com/verlab/accelerated_features.git', 'e92685f57f8318b18725c5c8c0bd28c7fe188d9a'),
    }
    for name, (url, commit) in versions.items():
        repo = root / name
        if not repo.exists():
            subprocess.run(['git', 'clone', url, str(repo)], check=True)
            subprocess.run(['git', '-C', str(repo), 'checkout', '--detach', commit], check=True)
        actual = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip()
        if actual != commit:
            raise RuntimeError(f'{repo} must be at {commit}; refusing to modify existing checkout')
    device = 'cuda'
    with torch.inference_mode():
        descriptor = MegaLocDescriptorExtractor(str(root / 'MegaLoc'), device=device)
        assert descriptor(torch.zeros(1, 3, 240, 384, device=device)).shape == (1, 8448)
        xfeat = XFeatFrontend(str(root / 'accelerated_features'), top_k=256, device=device)
        features = xfeat.detect(torch.rand(2, 3, 128, 160, device=device))
        assert xfeat.matcher({'image0': features[0], 'image1': features[1]})['matches'].shape[-1] == 2
    weights = Path('/models/dpvo.pth')
    vocabulary = Path('/models/ORBvoc.txt')
    if not vocabulary.is_file():
        raise RuntimeError('Copy ORBvoc.txt into /models first')
    (root / 'ready.json').write_text(json.dumps({
        'repos': versions, 'torch': torch.__version__,
        'checkpoint_sha256': hashlib.sha256(weights.read_bytes()).hexdigest()}, indent=2))
    print('Learned models cached and exercised; now build device-local TensorRT engines.')


if __name__ == '__main__':
    main()

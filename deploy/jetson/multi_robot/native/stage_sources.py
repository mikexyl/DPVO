"""Snapshot local CBS/GTSAM/aria_common sources for the native Docker build."""
import argparse
import json
from pathlib import Path
import subprocess
import tarfile

parser = argparse.ArgumentParser(__doc__)
parser.add_argument('--gtsam', type=Path, required=True)
parser.add_argument('--aria-common', type=Path, required=True)
parser.add_argument('--cbs', type=Path, default=Path(__file__).resolve().parents[4] / 'cbs')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
sources = {'gtsam': args.gtsam, 'aria_common': args.aria_common, 'cbs': args.cbs}
manifest = {}
for name, source in sources.items():
    if not (source / 'CMakeLists.txt').is_file():
        raise ValueError(f'Missing source: {source}')
    manifest[name] = {'commit': subprocess.check_output(
        ['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip(),
        'dirty': bool(subprocess.check_output(
            ['git', '-C', str(source), 'status', '--porcelain'], text=True).strip())}

def source_only(info):
    if any(p.startswith(('.git', 'build', '.pixi')) or p in
           ('datasets', 'results', '__pycache__') for p in Path(info.name).parts):
        return None
    return info

with tarfile.open(args.output, 'w:gz') as archive:
    for name, source in sources.items():
        archive.add(source, arcname=name, filter=source_only)
args.output.with_suffix('.manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
print(args.output)

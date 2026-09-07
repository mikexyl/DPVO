"""Fetch the pinned SPHORB tables into models/sphorb and verify every SHA-256."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'native/sphorb/tables.json').read_text())
    output = ROOT / 'models/sphorb'
    output.mkdir(parents=True, exist_ok=True)
    def valid(name, spec):
        path = output / name
        return path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == spec['sha256']
    if not all(valid(name, spec) for name, spec in manifest['files'].items()):
        revision = manifest['revision']
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / 'upstream.tar.gz'
            subprocess.run(['curl', '-fL', f'https://codeload.github.com/tdsuper/SPHORB/tar.gz/{revision}',
                            '-o', str(archive)], check=True)
            with tarfile.open(archive) as tar:
                for name, spec in manifest['files'].items():
                    data = tar.extractfile(f'SPHORB-{revision}/Data/{name}').read()
                    if len(data) != spec['bytes'] or hashlib.sha256(data).hexdigest() != spec['sha256']:
                        raise ValueError(f'Checksum mismatch: {name}')
                    (output / name).write_bytes(data)
    print(f'Verified {len(manifest["files"])} SPHORB tables in {output}')


if __name__ == '__main__':
    main()

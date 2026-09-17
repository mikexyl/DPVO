"""Bundle a trusted local CBS executable and its non-system shared libraries."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
import re
import shutil
import subprocess


def bundle(binary, output):
    binary, output = Path(binary).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError(f'Use a new output directory; refusing to overwrite {output}')
    report = subprocess.check_output(['ldd', str(binary)], text=True)
    if 'not found' in report:
        raise RuntimeError(report)
    libraries = []
    for line in report.splitlines():
        match = re.search(r'=> (/\S+)', line)
        if match and not re.match(r'lib(c|m|pthread|dl|rt|stdc\+\+|gcc_s)\.so', Path(match[1]).name):
            libraries.append(Path(match[1]))
    (output / 'bin').mkdir(parents=True)
    (output / 'lib').mkdir()
    shutil.copy2(binary, output / 'bin' / binary.name)
    for library in libraries:
        shutil.copy2(library, output / 'lib' / library.name)
    manifest = {'architecture': platform.machine(), 'source_binary': str(binary),
                'files': {str(f.relative_to(output)): hashlib.sha256(f.read_bytes()).hexdigest()
                          for f in output.rglob('*') if f.is_file()}}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--output', default=str(Path(__file__).parent / 'bundle'))
    args = parser.parse_args()
    print(json.dumps(bundle(args.binary, args.output), indent=2))

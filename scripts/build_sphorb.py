"""Build the optional GPL-derived CPU extension without changing Python packages."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
PYBIND_SHA = 'e08cb87f4773da97fa7b5f035de8763abc656d87d5773e62f6da0587d1f0ec20'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sanitize', action='store_true')
    args = parser.parse_args()
    build = ROOT / 'build' / ('sphorb-sanitize' if args.sanitize else 'sphorb')
    build.mkdir(parents=True, exist_ok=True)
    archive = ROOT / 'build' / 'sphorb-pybind11-2.13.6.tar.gz'
    if not archive.exists():
        subprocess.run(['curl', '-fL', 'https://codeload.github.com/pybind/pybind11/tar.gz/refs/tags/v2.13.6',
                        '-o', str(archive)], check=True)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != PYBIND_SHA:
        raise ValueError('pybind11 archive checksum mismatch')
    headers = ROOT / 'build' / 'pybind11-2.13.6'
    if not headers.exists():
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                if not (ROOT / 'build' / member.name).resolve().is_relative_to(ROOT / 'build') or member.issym() or member.islnk():
                    raise ValueError('Unsafe archive member')
            tar.extractall(ROOT / 'build')
    subprocess.run(['/usr/bin/cmake', '-S', str(ROOT / 'native/sphorb'), '-B', str(build),
                    '-DCMAKE_BUILD_TYPE=RelWithDebInfo', '-DCMAKE_CXX_COMPILER=/usr/bin/g++',
                    '-DCMAKE_CXX_FLAGS=', '-DCMAKE_EXE_LINKER_FLAGS=',
                    '-DCMAKE_SHARED_LINKER_FLAGS=', '-DCMAKE_MODULE_LINKER_FLAGS=',
                    f'-DPython_EXECUTABLE={sys.executable}', f'-DPYBIND11_SOURCE={headers}',
                    f'-DSPHORB_SANITIZE={"ON" if args.sanitize else "OFF"}'], check=True)
    # Sanitizer runs use a native executable; do not replace the normal Python module.
    target = 'sphorb_audit' if args.sanitize else 'all'
    subprocess.run(['/usr/bin/cmake', '--build', str(build), '--target', target, '--parallel', '4'], check=True)


if __name__ == '__main__':
    main()

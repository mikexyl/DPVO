"""Build optional TEASER++/PMC without changing installed Python packages."""
import hashlib
import argparse
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
SOURCES = (
    ('MIT-SPARK/TEASER-plusplus', '52a9c52ee7d4c838c5e8a75458c33178be5bfb70',
     'dc9ec391613b470b175267ca60c95099e74af818be3586cbb98ecd32cf806d59', 'TEASER_SOURCE'),
    ('jingnanshi/pmc', 'a2dfd612a501bca83c47206255dbbff619481f97',
     '34a2c9716c903ccff3b04358470fb6e3cf1b97d0f2badc534d4c003613084851', 'PMC_SOURCE'),
    ('pybind/pybind11', 'v2.13.6',
     'e08cb87f4773da97fa7b5f035de8763abc656d87d5773e62f6da0587d1f0ec20', 'PYBIND11_SOURCE'),
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sanitize', action='store_true')
    args = parser.parse_args()
    cache = ROOT / 'build/teaser-sources'
    cache.mkdir(parents=True, exist_ok=True)
    definitions = []
    for repo, rev, checksum, definition in SOURCES:
        archive = cache / f'{repo.split("/")[-1]}-{rev}.tar.gz'
        if not archive.exists():
            temporary = archive.with_suffix('.part')
            ref = f'refs/tags/{rev}' if repo == 'pybind/pybind11' else rev
            subprocess.run(['curl', '-fL', f'https://codeload.github.com/{repo}/tar.gz/{ref}',
                            '-o', str(temporary)], check=True)
            temporary.rename(archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != checksum:
            raise ValueError(f'Checksum mismatch: {archive}')
        with tarfile.open(archive) as tar:
            members = tar.getmembers()
            for member in members:
                if (not (cache/member.name).resolve().is_relative_to(cache)
                        or not (member.isfile() or member.isdir())):
                    raise ValueError(f'Unsafe archive member: {member.name}')
            # Re-extract verified sources so local stale edits cannot affect the build.
            tar.extractall(cache)
            source = cache / members[0].name.split('/')[0]
        definitions.append(f'-D{definition}={source}')
    build = ROOT / ('build/teaser-sanitize' if args.sanitize else 'build/teaser')
    subprocess.run(['/usr/bin/cmake', '-S', str(ROOT/'native/teaser'), '-B', str(build),
                    '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CXX_COMPILER=/usr/bin/g++',
                    '-DCMAKE_CXX_FLAGS=', '-DCMAKE_EXE_LINKER_FLAGS=',
                    '-DCMAKE_SHARED_LINKER_FLAGS=', '-DCMAKE_MODULE_LINKER_FLAGS=',
                    f'-DPython_EXECUTABLE={sys.executable}',
                    f'-DTEASER_SANITIZE={"ON" if args.sanitize else "OFF"}', *definitions], check=True)
    subprocess.run(['/usr/bin/cmake', '--build', str(build), '--target', 'teaser_audit' if args.sanitize else '_teaser',
                    '--parallel', '4'], check=True)


if __name__ == '__main__':
    main()

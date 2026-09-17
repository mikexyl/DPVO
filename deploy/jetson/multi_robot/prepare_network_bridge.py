"""Fetch a pinned, checksum-verified official bridge binary into ignored storage."""
import argparse
import hashlib
from pathlib import Path
import urllib.request
import zipfile

VERSION = '1.10.1'
DIGESTS = {
    'aarch64': 'fdb64d942d4b6beccbe9b1f8a359a01a2fca95a023c27535bf97b233603fad5f',
    'x86_64': 'f3ea8ac4345be22716925269bfd3e350e7cb36a8f056e835468c8299341d29b0',
}
if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--arch', choices=DIGESTS, required=True)
    args = parser.parse_args()
    output = Path(__file__).resolve().parent / 'network-bin'
    output.mkdir(exist_ok=True)
    name = f'zenoh-plugin-ros2dds-{VERSION}-{args.arch}-unknown-linux-gnu-standalone.zip'
    archive = output / name
    urllib.request.urlretrieve(f'https://github.com/eclipse-zenoh/zenoh-plugin-ros2dds/releases/download/{VERSION}/{name}', archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != DIGESTS[args.arch]:
        raise SystemExit('Bridge release checksum mismatch')
    with zipfile.ZipFile(archive) as z:
        binary = output / 'zenoh-bridge-ros2dds'
        binary.write_bytes(z.read('zenoh-bridge-ros2dds'))
        binary.chmod(0o755)
    print(binary)

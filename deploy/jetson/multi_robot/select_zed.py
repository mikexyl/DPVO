"""Locate only the selected ZED UVC capture node, leaving other sensors private."""
import argparse
from pathlib import Path


def select(serial, sys_root=Path('/sys')):
    usb_root = sys_root / 'bus/usb/devices'
    matches = []
    for video in (sys_root / 'class/video4linux').glob('*'):
        if (video / 'index').read_text().strip() != '0':
            continue
        camera = next((p for p in (video / 'device').resolve().parents
                       if (p / 'idVendor').exists()), None)
        if camera is None or (camera / 'idVendor').read_text().strip() != '2b03':
            continue
        # ZED 2 stores the serial on its HID sibling within the camera's hub.
        identities = [camera] + [p.resolve() for p in usb_root.glob('*')
                                 if p.resolve().parent == camera.parent]
        if serial and not any((p / 'serial').exists() and
                              (p / 'serial').read_text().strip() == serial for p in identities):
            continue
        matches.append('/dev/' + video.name)
    if len(matches) != 1:
        raise RuntimeError(f'Expected one ZED capture node for serial {serial}, found {matches}')
    return matches[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--serial', required=True)
    print(select(parser.parse_args().serial))

"""Host-side service helper: boost clocks only while the camera worker runs."""
import json
from pathlib import Path
import subprocess
import sys
import time


def main():
    status_file = Path(sys.argv[1])
    snapshot = sys.argv[2]
    started = time.time()
    boosted = False
    while True:
        try:
            active = (status_file.stat().st_mtime >= started and
                      json.loads(status_file.read_text()).get('state') in
                      {'starting', 'tracking', 'waiting_for_motion', 'tracking_lost'})
        except (OSError, ValueError):
            active = False
        if active != boosted:
            command = ['jetson_clocks'] if active else ['jetson_clocks', '--restore', snapshot]
            subprocess.run(command, check=True)
            boosted = active
        time.sleep(0.5)


if __name__ == '__main__':
    main()

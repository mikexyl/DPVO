"""Fix NVIDIA startup scripts that hard-code usb0 for both RNDIS and NCM.

Run as root on the Jetson. Keep the original script alongside the patched file.
No running service is restarted. Interface names come from configfs, so the fix
also works when only one Ethernet gadget function is enabled.
"""
from pathlib import Path
import shutil
import subprocess

path = Path('/opt/nvidia/l4t-usb-device-mode/nv-l4t-usb-device-mode-start.sh')
text = path.read_text()
patched = text
for flag, function in [('rndis', 'rndis'), ('ecm', '${ecm_ncm}')]:
    old = f'if [ ${{enable_{flag}}} -eq 1 ]; then\n    ifname="usb0"'
    new = (f'if [ ${{enable_{flag}}} -eq 1 ]; then\n'
           f'    ifname="$(cat /sys/kernel/config/usb_gadget/l4t/functions/{function}.usb0/ifname)"')
    if old in patched:
        patched = patched.replace(old, new, 1)
    elif new not in patched:
        raise SystemExit(f'Unrecognized {flag} configuration; no changes made')
if patched != text:
    subprocess.run(['bash', '-n'], input=patched, text=True, check=True)
    backup = path.with_name(path.name + '.before-dpvo-interface-fix')
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(patched)
    print(f'Fixed USB interface discovery; backup: {backup}')
else:
    print('USB interface discovery already fixed')

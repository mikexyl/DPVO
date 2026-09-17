"""Human-readable fleet health, using monotonic receive ages rather than wall clocks."""
import math


def compact_health_text(status, receive_age):
    """Keep failures visible without repeating the diagnostic telemetry."""
    if status is None:
        return '⚪ Waiting'
    if receive_age > 3:
        return '🔴 Offline'
    if status.get('state') != 'worker_running':
        code = status.get('exit_code')
        return f'🔴 Stopped · exit {code}' if code not in (None, 0) else '🟢 Online · Idle'
    for name, key in (('Camera', 'camera_age_s'), ('Tracker', 'tracker_age_s')):
        age = status.get(key)
        if age is not None and (not math.isfinite(float(age)) or float(age) + receive_age >= 3):
            return f'🟡 {name} stalled'
    tracking = status.get('tracking') or {}
    if tracking.get('state') == 'tracking':
        return f"🟢 Tracking · {tracking.get('processing_fps', 0):.1f} FPS"
    return '🟡 Initializing · move slowly'


def health_text(status, receive_age):
    if status is None:
        return '⚪ Waiting for heartbeat'
    if receive_age > 3:
        return f'🔴 Disconnected · last heartbeat {receive_age:.1f}s ago'
    sequence = status.get('heartbeat')
    pulse = '🟢' if sequence is None or int(sequence) % 2 else '💚'
    text = f'{pulse} Online · heartbeat {sequence if sequence is not None else "received"} · {receive_age:.1f}s ago'
    if status.get('state') != 'worker_running':
        text += '\n\nCamera off · tracker stopped'
        if status.get('exit_code') not in (None, 0):
            text += f" · error (exit {status['exit_code']})"
        return text
    if 'camera_age_s' not in status and 'tracker_age_s' not in status:
        return text + '\n\nWorker running · detailed frame telemetry unavailable on this robot'
    def activity(name, key):
        age = status.get(key)
        if age is None:
            return f'{name}: waiting for first frame'
        age = float(age) + receive_age
        return f'{name}: ' + ('active' if math.isfinite(age) and age < 3 else f'stalled ({age:.1f}s)')
    text += '\n\n' + activity('Camera', 'camera_age_s') + ' · ' + activity('Tracker', 'tracker_age_s')
    tracking = status.get('tracking') or {}
    if tracking:
        state = 'Tracking' if tracking.get('state') == 'tracking' else 'Initializing — move slowly'
        text += (f"\n\n{state} · {tracking.get('processing_fps', 0):.1f} FPS"
                 f" · {tracking.get('keyframes', 0)} keyframes"
                 f" · {tracking.get('patch_tracks', 0)} patches")
    return text

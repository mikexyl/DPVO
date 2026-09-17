import sys
from pathlib import Path
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ros2/dpvo_multi_robot'))
from dpvo_multi_robot.online_status import health_text, compact_health_text


class FleetHealthTest(unittest.TestCase):
    def test_compact_status_keeps_failures_visible(self):
        self.assertIn('Waiting', compact_health_text(None, 0))
        status = dict(state='stopped', exit_code=1)
        self.assertIn('exit 1', compact_health_text(status, 0))
        self.assertIn('Offline', compact_health_text(status, 4))
        status.update(exit_code=0)
        self.assertIn('Idle', compact_health_text(status, 0))
        status.update(state='worker_running', camera_age_s=.1, tracker_age_s=3.1,
                      tracking=dict(state='tracking', processing_fps=9))
        self.assertIn('Tracker stalled', compact_health_text(status, 0))
        status['tracker_age_s'] = .1
        self.assertIn('Tracking · 9.0 FPS', compact_health_text(status, 0))
        status['camera_age_s'] = float('nan')
        self.assertIn('Camera stalled', compact_health_text(status, 0))

    def test_idle_online_disconnected_and_stall_are_distinct(self):
        self.assertIn('Waiting', health_text(None, 0))
        status = dict(state='stopped', heartbeat=2)
        self.assertIn('Camera off', health_text(status, .1))
        self.assertIn('Disconnected', health_text(status, 4))
        status.update(state='worker_running', camera_age_s=.1, tracker_age_s=4)
        self.assertIn('Camera: active', health_text(status, .2))
        self.assertIn('Tracker: stalled', health_text(status, .2))
        status.update(tracker_age_s=.2, tracking=dict(state='initializing', processing_fps=9))
        self.assertIn('Initializing', health_text(status, 0))
        self.assertIn('9.0 FPS', health_text(status, 0))
        status['tracking']['state'] = 'tracking'
        self.assertIn('Tracking', health_text(status, 0))


if __name__ == '__main__':
    unittest.main()

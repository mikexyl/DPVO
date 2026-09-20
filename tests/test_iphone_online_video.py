"""Causal map deformation and render checks, without ROS or a GPU."""
import importlib.util
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("iphone_video", ROOT / "deploy/blackwell_ros2/iphone_online_video.py")
VIDEO = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VIDEO)

try:
    from dpvo_multi_robot.online_cbs import OnlineCbsNode
    from dpvo_multi_robot.cbs_pgo import CbsPgoNode
    CBS_AVAILABLE = True
except ImportError:
    CBS_AVAILABLE = False


@unittest.skipUnless(CBS_AVAILABLE, "Source ROS and its interfaces overlay")
class OnlineGroupsTest(unittest.TestCase):
    def test_component_path_publication_does_not_clear_other_robots(self):
        from builtin_interfaces.msg import Time
        node = object.__new__(CbsPgoNode)
        node.robot_ids = ['r0', 'r1', 'r2']
        node.get_clock = Mock()
        node.get_clock.return_value.now.return_value.to_msg.return_value = Time()
        node.get_parameter = Mock(return_value=SimpleNamespace(value='world'))
        publishers = {r: Mock() for r in node.robot_ids}
        node._publish_paths({('r0', '001', 0): VIDEO.Sim3.identity()}, publishers)
        publishers['r0'].publish.assert_called_once()
        publishers['r1'].publish.assert_not_called()
        publishers['r2'].publish.assert_not_called()

    def test_disconnected_groups_have_separate_immutable_snapshots(self):
        node = object.__new__(OnlineCbsNode)
        constraints = [SimpleNamespace(query_robot=('r0', '001'), match_robot=('r1', '001')),
                       SimpleNamespace(query_robot=('r2', '001'), match_robot=('r3', '001'))]
        paths = {f'r{i}': [i] for i in range(4)}
        sessions = {r: '001' for r in paths}
        with patch.object(CbsPgoNode, '_run') as solver:
            node._run(constraints, paths, sessions, {}, {r: [0.] for r in paths})
        self.assertEqual(solver.call_count, 2)
        self.assertEqual([set(c.args[1]) for c in solver.call_args_list],
                         [{'r0', 'r1'}, {'r2', 'r3'}])
        self.assertTrue(all(len(c.args[0]) == 1 for c in solver.call_args_list))


class LiveMapTest(unittest.TestCase):
    def test_viewports_are_disjoint_and_full_map_uses_whole_area(self):
        for sizes in ([1]*14, [10, 2, 1, 1], [14]):
            groups = [(i, list(range(n))) for i, n in enumerate(sizes)]
            rectangles = [rect for _, rect in VIDEO.map_viewports(groups)]
            for i, (x, y, w, h) in enumerate(rectangles):
                self.assertTrue(22 <= x < x+w <= 1422 and 134 <= y < y+h <= 984)
                for a, b, c, d in rectangles[i+1:]:
                    self.assertTrue(x+w <= a or a+c <= x or y+h <= b or b+d <= y)
            if len(sizes) == 1:
                self.assertEqual(rectangles, [(22, 134, 1400, 850)])

    def test_display_filter_removes_far_depth_rays_without_changing_recording(self):
        rng = np.random.default_rng(3)
        points = np.vstack([rng.normal(size=(100, 3)), [[10000, 0, 10000]]])
        original = points.copy()
        visible = VIDEO.display_cloud(points)
        self.assertEqual(len(visible), 100)
        np.testing.assert_array_equal(points, original)

    def test_no_completed_solution_preserves_raw_geometry(self):
        poses = np.array([[0, 0, 0, 0, 0, 0, 1], [2, 0, 0, 0, 0, 0, 1.]])
        points = np.array([[0., 0, 1], [2., 0, 1]])
        result, path = VIDEO.warp_live_map(points, np.array([0, 1]), poses, [10., 20.], {}, VIDEO.Sim3.identity())
        np.testing.assert_allclose(result, points)
        np.testing.assert_allclose(path, poses[:, :3])

    def test_landmarks_follow_own_keyframes_and_unseen_frames_use_anchor(self):
        poses = np.array([[0, 0, 0, 0, 0, 0, 1], [2, 0, 0, 0, 0, 0, 1.]])
        points = np.array([[0., 0, 1], [2., 0, 1]])
        anchor = VIDEO.Sim3([10, 0, 0], np.eye(3), 2.)
        optimized = {10.: VIDEO.Sim3([11, 0, 0], np.eye(3), 3.)}
        result, path = VIDEO.warp_live_map(points, np.array([0, 1]), poses, [10., 20.], optimized, anchor)
        np.testing.assert_allclose(result, [[11, 0, 3], [14, 0, 2]])
        np.testing.assert_allclose(path, [[11, 0, 0], [14, 0, 0]])

    def test_keyframe_reindexing_does_not_reassign_old_solution(self):
        poses = np.array([[1, 0, 0, 0, 0, 0, 1.]])
        optimized = {10.: VIDEO.Sim3([100, 0, 0], np.eye(3), 1.)}
        result, path = VIDEO.warp_live_map(np.array([[1., 0, 1]]), np.array([0]), poses,
                                          [20.], optimized, VIDEO.Sim3.identity())
        np.testing.assert_allclose(result, [[1, 0, 1]])
        np.testing.assert_allclose(path, [[1, 0, 0]])


if __name__ == '__main__':
    unittest.main()

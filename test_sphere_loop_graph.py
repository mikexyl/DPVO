import copy
import unittest

import numpy as np

from plot_sphere_loop_graph import build_graph


class SphereLoopGraphTests(unittest.TestCase):
    def setUp(self):
        self.trajectory = np.zeros((6, 8))
        self.trajectory[:, 0] = [0, 10, 20, 30, 40, 50]
        self.trajectory[:, 1] = np.arange(6) * 2
        self.trajectory[:, 7] = 1
        self.keyframes = dict(source_timestamps=[0, 10, 30, 50])
        self.manifest = dict(snapshots=[dict(anchor_timestamp=10, source_timestamps=[0, 10], center=[999, 0, 0]),
                                        dict(anchor_timestamp=50, source_timestamps=[30, 50], center=[999, 0, 0])])
        self.report = dict(queries=[dict(query_anchor=50, candidates=[dict(anchor=10, score=.4,
                                            mutual_matches=6, geometric_inliers=0)])])

    def test_final_pose_timestamp_mapping_and_candidate_status(self):
        graph = build_graph(self.trajectory, self.keyframes, self.manifest, self.report)
        self.assertEqual([n['position'][0] for n in graph['nodes']], [0, 2, 6, 10])
        self.assertEqual(graph['sequential_edges'], [(0, 10), (10, 30), (30, 50)])
        self.assertEqual(graph['candidate_edges'][0]['status'], 'unverified_appearance_candidate')
        self.assertTrue(graph['visualization_only'])

    def test_diagnostic_inliers_do_not_promote_to_confirmed_loop(self):
        self.report['queries'][0]['candidates'][0]['geometric_inliers'] = 20
        graph = build_graph(self.trajectory, self.keyframes, self.manifest, self.report)
        self.assertEqual(graph['candidate_edges'][0]['status'], 'unverified_appearance_candidate')

    def test_no_candidates(self):
        graph = build_graph(self.trajectory, self.keyframes, self.manifest, dict(queries=[]))
        self.assertEqual(graph['candidate_edges'], [])

    def test_missing_poses_overlap_and_invalid_edges_rejected(self):
        with self.assertRaisesRegex(ValueError, 'missing'):
            build_graph(self.trajectory[:-1], self.keyframes, self.manifest, self.report)
        manifest = copy.deepcopy(self.manifest)
        manifest['snapshots'][1]['source_timestamps'].append(10)
        with self.assertRaisesRegex(ValueError, 'share'):
            build_graph(self.trajectory, self.keyframes, manifest, self.report)
        for candidate in (dict(anchor=50, score=.4), dict(anchor=10, score=float('nan'))):
            self.report['queries'][0]['candidates'][0].update(candidate)
            with self.assertRaises(ValueError):
                build_graph(self.trajectory, self.keyframes, self.manifest, self.report)


if __name__ == '__main__':
    unittest.main()

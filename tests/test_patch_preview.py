import unittest
from types import SimpleNamespace as NS
import numpy as np
import torch
from deploy.jetson.patch_preview import PatchPreview


class PatchPreviewTest(unittest.TestCase):
    def test_trails_bounded_invalid_points_and_candidates_reset(self):
        preview = PatchPreview(max_tracks=2, trail_length=3)
        image = np.zeros((100, 100, 3), np.uint8)
        for x in range(5):
            drawn, count = preview.draw(image, [(10, 2), (10, 3), (10, 4)],
                                       [[40+x, 50], [np.nan, 40], [50, 50]])
        self.assertEqual(count, 1)
        self.assertEqual(len(preview.history[(10, 2)]), 3)
        self.assertTrue(drawn.any())
        self.assertFalse(image.any())
        preview.draw(image, [(11, 2)], [[50, 50]], tracked=False)
        self.assertFalse(preview.history)

    def test_reprojections_use_current_target_and_stable_patch_identity(self):
        graph = NS(tstamps_=np.array([14, 20, 30]), ii=torch.tensor([0, 1, 2]),
                   jj=torch.tensor([2, 1, 2]), kk=torch.tensor([3, 5, 8]))
        def reproject(indices):
            self.assertEqual(indices[0].tolist(), [0])
            return torch.full((1, 1, 2, 3, 3), 10.)
        slam = NS(n=3, pg=graph, is_initialized=True, M=4, P=3, RES=4, reproject=reproject)
        preview = PatchPreview()
        _, count = preview.snapshot(slam, np.zeros((100, 100, 3), np.uint8), 30)
        self.assertEqual(count, 1)
        self.assertEqual(list(preview.history), [(14, 3)])
        self.assertEqual(preview.history[(14, 3)][-1], (40, 40))

    def test_retains_existing_tracks_and_observes_between_rendered_frames(self):
        graph = NS(tstamps_=np.array([14, 20, 30]), ii=torch.tensor([0, 1]),
                   jj=torch.tensor([2, 2]), kk=torch.tensor([3, 5]))
        location = [10.]
        selected = []
        def reproject(indices):
            selected.append(indices[0].tolist())
            return torch.full((1, 1, 2, 3, 3), location[0])
        slam = NS(n=3, pg=graph, is_initialized=True, M=4, P=3, RES=4, reproject=reproject)
        preview = PatchPreview(max_tracks=1)
        preview.draw(np.zeros((100, 100, 3), np.uint8), [(14, 3)], [[30, 30]])
        frame = np.zeros((100, 100, 3), np.uint8)
        result, _ = preview.snapshot(slam, frame, 30, render=False)
        self.assertIsNone(result)
        location[0] = 12.
        preview.snapshot(slam, frame, 30)
        self.assertEqual(selected, [[0], [0]])
        self.assertEqual(list(preview.history[(14, 3)]), [(30, 30), (40, 40), (48, 48)])

    def test_only_one_source_keyframe_and_switch_when_retired(self):
        graph = NS(tstamps_=np.array([14, 20, 30]), ii=torch.tensor([0, 1, 1]),
                   jj=torch.tensor([2, 2, 2]), kk=torch.tensor([3, 4, 5]))
        selected = []
        def reproject(indices):
            selected.append(indices[0].tolist())
            return torch.full((1, len(indices[0]), 2, 3, 3), 10.)
        slam = NS(n=3, pg=graph, is_initialized=True, M=4, P=3, RES=4,
                  reproject=reproject)
        preview = PatchPreview()
        frame = np.zeros((100, 100, 3), np.uint8)
        preview.snapshot(slam, frame, 30)
        self.assertEqual(selected[-1], [1, 1])
        self.assertEqual({identity[0] for identity in preview.history}, {20})
        graph.ii, graph.jj, graph.kk = torch.tensor([0]), torch.tensor([2]), torch.tensor([3])
        preview.snapshot(slam, frame, 30)
        self.assertEqual(selected[-1], [0])
        self.assertEqual({identity[0] for identity in preview.history}, {14})
        self.assertEqual(len(preview.history[(14, 3)]), 1)

    def test_rejected_initialization_frame_uses_candidate_slot(self):
        patches = torch.zeros(3, 4, 3, 3, 3)
        patches[2, :, :2] = 10
        graph = NS(tstamps_=np.array([10, 11, 12]), patches_=patches)
        slam = NS(n=2, pg=graph, is_initialized=False, P=3, RES=4)
        preview = PatchPreview()
        _, count = preview.snapshot(slam, np.zeros((100, 100, 3), np.uint8), 12)
        self.assertEqual(count, 4)
        self.assertFalse(preview.history)


if __name__ == '__main__':
    unittest.main()

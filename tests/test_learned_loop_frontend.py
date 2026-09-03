import unittest
from unittest import mock

import numpy as np
import torch

from dpvo.loop_closure.learned_frontend import (
    MegaLocDescriptorExtractor,
    XFeatFrontend,
)


class _FakeMegaLoc(torch.nn.Module):
    def forward(self, images):
        self.images = images.detach().clone()
        return images.mean(dim=(-1, -2))


class _FakeXFeat(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dev = torch.device("cpu")

    def detectAndCompute(self, images, top_k):
        del top_k
        return [
            {
                "keypoints": torch.tensor(
                    [[10.0, 20.0], [30.0, 40.0]],
                    device=images.device,
                ),
                "descriptors": torch.eye(
                    2,
                    64,
                    device=images.device,
                ),
            }
            for _ in images
        ]

    def match_lighterglue(self, first, second, min_conf):
        self.last_min_conf = min_conf
        indices = np.array([[0, 1], [1, 0]], dtype=np.int64)
        return (
            first["keypoints"][indices[:, 0]].numpy(),
            second["keypoints"][indices[:, 1]].numpy(),
            indices,
        )


class LearnedLoopFrontendTest(unittest.TestCase):
    def test_megaloc_applies_rgb_imagenet_preprocessing_and_normalizes(self):
        model = _FakeMegaLoc()
        with mock.patch(
            "dpvo.loop_closure.learned_frontend._load_hub_model",
            return_value=model,
        ):
            extractor = MegaLocDescriptorExtractor("unused", device="cpu")

        bgr = torch.tensor([10, 20, 30], dtype=torch.uint8).view(1, 3, 1, 1)
        descriptor = extractor(bgr)

        expected_pixel = torch.tensor(
            [
                (30.0 / 255.0 - 0.485) / 0.229,
                (20.0 / 255.0 - 0.456) / 0.224,
                (10.0 / 255.0 - 0.406) / 0.225,
            ]
        )
        torch.testing.assert_close(model.images[0, :, 0, 0], expected_pixel)
        torch.testing.assert_close(descriptor.norm(dim=-1), torch.ones(1))

    def test_xfeat_features_and_lighterglue_match_indices_are_compatible(self):
        model = _FakeXFeat()
        with mock.patch(
            "dpvo.loop_closure.learned_frontend._load_hub_model",
            return_value=model,
        ):
            frontend = XFeatFrontend(
                "unused",
                top_k=2,
                min_confidence=0.2,
                device="cpu",
            )

        features = frontend.detect(torch.rand(2, 3, 64, 96))
        result = frontend.matcher(
            {"image0": features[0], "image1": features[1]}
        )

        self.assertEqual(features[0]["descriptors"].shape, (1, 2, 64))
        torch.testing.assert_close(
            features[0]["image_size"],
            torch.tensor([[96.0, 64.0]]),
        )
        torch.testing.assert_close(
            result["matches"],
            torch.tensor([[[0, 1], [1, 0]]]),
        )
        self.assertEqual(model.last_min_conf, 0.2)


if __name__ == "__main__":
    unittest.main()

"""Hybrid graph math and fallback tests; no TensorRT installation required."""
import unittest
from unittest.mock import patch

import torch
from torch import nn
from dpvo.net import Update
from deploy.jetson.tensorrt_update import TensorRTUpdate, UpdateHead, UpdateTail


class TupleHead(UpdateHead):
    def forward(self, *args):
        return (super().forward(*args),)


class TensorRTUpdateTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        self.original = Update(3).eval()
        self.hybrid = TensorRTUpdate.__new__(TensorRTUpdate)
        nn.Module.__init__(self.hybrid)
        self.hybrid.original = self.original
        self.hybrid.head = TupleHead(self.original)
        self.hybrid.tail = UpdateTail(self.original)
        self.hybrid.max_edges, self.hybrid.fallback_calls = 64, 0
        n = 8
        self.inputs = (torch.randn(1,n,384), torch.randn(1,n,384), torch.randn(1,n,882),
                       None, torch.arange(n)//4, torch.arange(n)%4, torch.arange(n)//2)
        self.neighbors = (torch.tensor([-1,0,1,2,3,4,5,6]), torch.tensor([1,2,3,4,5,6,7,-1]))

    def compare(self):
        with torch.no_grad(), patch('dpvo.net.fastba.neighbors', return_value=self.neighbors):
            actual = self.hybrid(*self.inputs)
            expected = self.original(*self.inputs)
        for a,b in zip((actual[0], *actual[1][:2]), (expected[0], *expected[1][:2])):
            torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_preserves_sequential_neighbor_and_group_aggregation(self):
        self.compare()
        self.assertEqual(self.hybrid.fallback_calls, 0)

    def test_out_of_profile_falls_back_explicitly(self):
        self.hybrid.max_edges = 4
        self.compare()
        self.assertEqual(self.hybrid.fallback_calls, 1)

    def test_rejects_training(self):
        with self.assertRaisesRegex(RuntimeError, 'inference only'):
            self.hybrid(*self.inputs)


if __name__ == '__main__': unittest.main()

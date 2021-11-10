import operator
import unittest

import torch
from numpy.testing import assert_array_compare

from src.losses.spatial_consistency_loss import spatial_consistency_loss


class ConsistencyLossTest(unittest.TestCase):

    """

        Key-points are in (N, T, K, 2)

    """

    def setUp(self) -> None:
        self.batch_size = 1
        self.n_key_points = 1

    def create_fake_coords(self) -> (torch.Tensor, torch.Tensor):
        close_fake_coords = torch.zeros(size=(self.batch_size, 5, self.n_key_points, 2))
        close_fake_coords[:, 1, ...] += 0.1
        close_fake_coords[:, 2, ...] += 0.2
        close_fake_coords[:, 3, ...] += 0.3
        close_fake_coords[:, 4, ...] += 0.4

        far_fake_coords = torch.zeros(size=(self.batch_size, 5, self.n_key_points, 2))
        far_fake_coords[:, 1, ...] += 0.1
        far_fake_coords[:, 2, ...] += 0.3
        far_fake_coords[:, 3, ...] += 0.4
        far_fake_coords[:, 4, ...] += 0.9

        far_far_fake_coords = torch.zeros(size=(self.batch_size, 5, self.n_key_points, 2))
        far_far_fake_coords[:, 1, ...] += 0.1
        far_far_fake_coords[:, 2, ...] += 0.1
        far_far_fake_coords[:, 3, ...] += 0.9
        far_far_fake_coords[:, 4, ...] += 0.9

        return close_fake_coords, far_fake_coords, far_far_fake_coords

    def assert_array_smaller(self, x, y):
        assert_array_compare(operator.__lt__, x, y, err_msg='',
                             verbose=True,
                             header='x not < y',
                             equal_inf=False)

    def test_close_vs_far(self):
        close_kpts, far_kpts, _ = self.create_fake_coords()
        L_close = spatial_consistency_loss(close_kpts).cpu().numpy()
        L_far = spatial_consistency_loss(far_kpts).cpu().numpy()
        assert L_close < L_far

    def test_close_vs_far_far(self):
        close_kpts, _, far_far_kpts = self.create_fake_coords()
        L_close = spatial_consistency_loss(close_kpts).cpu().numpy()
        L_far_far = spatial_consistency_loss(far_far_kpts).cpu().numpy()
        assert L_close < L_far_far

    def test_far_vs_far_far(self):
        _, far_kpts, far_far_kpts = self.create_fake_coords()
        L_far = spatial_consistency_loss(far_kpts).cpu().numpy()
        L_far_far = spatial_consistency_loss(far_far_kpts).cpu().numpy()
        assert L_far < L_far_far

    def test_close_vs_no_movement(self):
        close_kpts, _, _ = self.create_fake_coords()
        no_mvmt_kpts = torch.zeros(size=(self.batch_size, 5, self.n_key_points, 2))
        no_mvmt_kpts[...] = 0.1
        L_close = spatial_consistency_loss(close_kpts).cpu().numpy()
        L_no_mvmt = spatial_consistency_loss(no_mvmt_kpts).cpu().numpy()
        assert abs(L_close - L_no_mvmt) <= 0.01

    def test_grad_flow(self):
        close_kpts, far_kpts, _ = self.create_fake_coords()
        fake_net = torch.nn.Linear(in_features=torch.flatten(close_kpts).shape[0],
                                   out_features=torch.flatten(close_kpts).shape[0])
        close_kpts = fake_net(torch.flatten(close_kpts)).view(close_kpts.shape)

        L = torch.norm(close_kpts, p=2)

        L.backward()

        assert fake_net.weight.grad is not None


if __name__ == "__main__":
    unittest.main()

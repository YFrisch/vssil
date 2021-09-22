import operator
import unittest

import torch
import numpy as np
from numpy.testing import assert_array_compare

from src.losses.spatial_consistency_loss import spatial_consistency_loss

class ConsistencyLossTest(unittest.TestCase):

    def setUp(self) -> None:
        self.batch_size = 4
        self.n_key_points = 3

    def create_fake_coords(self) -> (torch.Tensor, torch.Tensor):
        close_fake_coords = torch.zeros(size=(self.batch_size, 5, self.n_key_points, 3))
        close_fake_coords[:, 1, ...] += 0.1
        close_fake_coords[:, 2, ...] += 0.2
        close_fake_coords[:, 3, ...] += 0.3
        close_fake_coords[:, 4, ...] += 0.6

        far_fake_coords = torch.zeros(size=(self.batch_size, 5, self.n_key_points, 3))
        far_fake_coords[:, 1, ...] += 0.1
        far_fake_coords[:, 2, ...] += 0.3
        far_fake_coords[:, 3, ...] += 0.6
        far_fake_coords[:, 4, ...] += 0.9

        return close_fake_coords, far_fake_coords

    def assert_array_smaller(self, x, y):
        assert_array_compare(operator.__lt__, x, y, err_msg='',
                             verbose=True,
                             header='x not < y',
                             equal_inf=False)

    def test_close_vs_far(self):
        close_k, far_k = self.create_fake_coords()
        np.testing.assert_array_less(
            x=spatial_consistency_loss(close_k, {}),
            y=spatial_consistency_loss(far_k, {})
        )


import os
from os.path import join
import yaml
import unittest

import torch
import numpy as np

from src.losses.temporal_separation_loss import temporal_separation_loss


class LossesTest(unittest.TestCase):

    def setUp(self) -> None:
        cwd = os.getcwd()
        with open(join(cwd, 'src/configs/ulosd.yml')) as cfg_file:
            self.cfg = yaml.safe_load(cfg_file)
        self.cfg['model']['n_feature_maps'] = 3
        self.cfg['model']['n_frames'] = 4

    def create_parallel_coords(self) -> np.ndarray:
        """ Creates 3 key-points moving along straight, parallel trajectories. """
        self.cfg['training']['separation_loss_sigma'] = 0.01
        num_timesteps = self.cfg['model']['n_frames']

        # Create three points
        coords = np.array(
            [[0, 0], [0, 1], [1, 0]],
            dtype=np.float32
        )

        # Expand in time
        coords = np.stack([coords] * num_timesteps, axis=0)

        # Add linear motion (Identical for all three points)
        coords += np.linspace(-1, 1, num_timesteps)[:, np.newaxis, np.newaxis]
        return coords[np.newaxis, ...]

    def test_temporal_separation_loss_parallel_movement(self):
        """ The loss should be high for parallel-moving key-points. """
        np_coords = self.create_parallel_coords()
        torch_coords = torch.tensor(np_coords)
        loss = temporal_separation_loss(self.cfg, torch_coords).cpu().numpy()
        np.testing.assert_almost_equal(loss, 1.0, decimal=4)

    def test_temporal_separation_loss_different_movement(self):
        """ the loss should be low for non-parallel movements. """
        # Create trajectories in which all key-points move differently
        np_coords = self.create_parallel_coords()
        np_coords[:, 0, :] = -np_coords[:, 0, :]
        np_coords[:, 1, :] = 0.0
        loss = temporal_separation_loss(self.cfg, torch.tensor(np_coords))
        np.testing.assert_almost_equal(loss, 0.0, decimal=4)


if __name__ == "__main__":
    unittest.main()
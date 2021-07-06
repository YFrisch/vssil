import unittest

import torch
import numpy as np

from src.models.ulosd_layers import add_coord_channels, FeatureMapsToKeyPoints,\
    KeyPointsToFeatureMaps


class OpsTest(unittest.TestCase):

    def test_add_coord_channel(self):
        B, C, H, W = 2, 3, 32, 32
        image_tensor = torch.zeros((B, C, H, W))
        extended_tensor = add_coord_channels(image_tensor)
        self.assertEqual(tuple(extended_tensor.shape), (B, C + 2, H, W))


class MapsToKeyPointsTest(unittest.TestCase):

    def setUp(self) -> None:
        self.map_shape = (1, 1, 33, 33)
        self.maps_2_keypoints = FeatureMapsToKeyPoints()

    def compute_coordinates(self, test_map: torch.Tensor) -> torch.Tensor:
        return self.maps_2_keypoints(test_map).squeeze()

    def test_zero_map_is_zero_coords(self):
        test_map = torch.zeros(self.map_shape)
        np.testing.assert_almost_equal(
            actual=self.compute_coordinates(test_map),
            desired=[0.0, 0.0, 0.0]
        )

    def test_object_in_top_left(self):
        test_map = torch.zeros(self.map_shape)
        test_map[0, 0, 0, 0] = 1.0
        np.testing.assert_almost_equal(
            actual=self.compute_coordinates(test_map),
            desired=[-1.0, 1.0, 1.0],
            decimal=2
        )

    def test_object_in_bottom_right(self):
        test_map = torch.zeros(self.map_shape)
        test_map[0, 0, -1, -1] = 1.0
        np.testing.assert_almost_equal(
            actual=self.compute_coordinates(test_map),
            desired=[1.0, -1.0, 1.0],
            decimal=2
        )

    def test_object_in_center(self):
        test_map = torch.zeros(self.map_shape)
        test_map[0, 0, self.map_shape[2]//2, self.map_shape[3]//2] = 1.0
        np.testing.assert_almost_equal(
            actual=self.compute_coordinates(test_map),
            desired=[0.0, 0.0, 1.0],
            decimal=2
        )


class KeyPointsToMapsTest(unittest.TestCase):

    def setUp(self) -> None:
        self.heatmap_width = 17
        self.keypoints_2_maps = KeyPointsToFeatureMaps(
            sigma=2.0,
            heatmap_width=self.heatmap_width
        )

    def compute_map(self, test_coordinates):
        test_coordinates = torch.tensor(test_coordinates).view((1, 1, -1))
        return self.keypoints_2_maps(test_coordinates).squeeze()

    def test_zero_scale_is_zero_map(self):
        np.testing.assert_array_equal(
            x=self.compute_map([0.0, 0.0, 0.0]),
            y=0.0
        )

    def test_object_in_top_left(self):
        test_map = self.compute_map([-1.0, 1.0, 1.0]).cpu().numpy()
        arg_max = np.concatenate((test_map == np.max(test_map)).nonzero())
        np.testing.assert_array_equal(
            x=arg_max,
            y=[0.0, 0.0]
        )

    def test_object_in_bottom_right(self):
        test_map = self.compute_map([1.0, -1.0, 1.0]).cpu().numpy()
        arg_max = np.concatenate((test_map == np.max(test_map)).nonzero())
        np.testing.assert_array_equal(
            x=arg_max,
            y=[self.heatmap_width-1, self.heatmap_width-1]
        )

    def test_object_in_center(self):
        test_map = self.compute_map([0.0, 0.0, 1.0]).cpu().numpy()
        arg_max = np.concatenate((test_map == np.max(test_map)).nonzero())
        np.testing.assert_array_equal(
            x=arg_max,
            y=[self.heatmap_width//2, self.heatmap_width//2]
        )


if __name__ == "__main__":
    unittest.main()

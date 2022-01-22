import unittest

import torch
import numpy as np

from src.models.ulosd_layers import add_coord_channels, FeatureMapsToKeyPoints,\
    KeyPointsToFeatureMaps
from src.models.ulosd_encoders import make_encoder, make_appearance_encoder
from src.models.ulosd_decoder import make_decoder


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


class VisionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.config = {
            'model': {
                'n_init_filters': 32,
                'n_convolutions_per_res': 1,
                'conv_kernel_size': 3,
                'feature_map_width': 16,
                'feature_map_height': 16,
                'n_feature_maps': 3,
                'encoder_hidden_activations': 'LeakyReLU',
                'decoder_hidden_activations': 'LeakyReLU',
            },
            'training': {
                'batch_size': 4
            }
        }
        self.input_shape = (self.config['training']['batch_size'], 2, 3, 64, 64)  # (N, T, C, H, W)
        self.enc, self.enc_inp_shape, self.enc_out_ch = make_encoder(self.input_shape, self.config)
        self.dec = make_decoder(self.enc_inp_shape, self.enc_out_ch, self.config)
        self.app_enc = make_appearance_encoder(self.input_shape, self.config)
        self.fmaps2kpts = FeatureMapsToKeyPoints()
        self.kpts2gmaps = KeyPointsToFeatureMaps(sigma=1.5, heatmap_width=16)

    def test_image_encoder_shapes(self):
        images = torch.zeros((self.config['training']['batch_size'], 3 + 2, 64, 64))
        kpt_feature_maps = self.enc(images)
        np.testing.assert_array_equal(
            x=tuple(kpt_feature_maps.shape),
            y=(self.config['training']['batch_size'], self.config['model']['n_feature_maps'],
               self.config['model']['feature_map_height'], self.config['model']['feature_map_width'])
        )

    def test_fmaps_to_kpts_shapes(self):
        images = torch.zeros(self.config['training']['batch_size'], 3 + 2, 64, 64)
        kpt_heatmaps = self.enc(images)
        kpt_coordinates = self.fmaps2kpts(kpt_heatmaps)
        np.testing.assert_array_equal(
            x=tuple(kpt_coordinates.shape),
            y=(self.config['training']['batch_size'], self.config['model']['n_feature_maps'], 3)
        )

    def test_kpts_to_gmaps_shapes(self):
        kpts = torch.zeros(self.config['training']['batch_size'],
                           self.config['model']['n_feature_maps'],
                           3)
        gmaps = self.kpts2gmaps(kpts)
        np.testing.assert_array_equal(
            x=tuple(gmaps.shape),
            y=(self.config['training']['batch_size'], self.config['model']['n_feature_maps'],
               self.config['model']['feature_map_height'], self.config['model']['feature_map_width'])
        )

    def test_image_decoder_shapes(self):

        first_frame = torch.zeros(self.config['training']['batch_size'],
                                  3, 64, 64)
        first_frame_fmaps = self.app_enc(first_frame)
        first_frame_kpts = torch.zeros(self.config['training']['batch_size'],
                                       self.config['model']['n_feature_maps'], 3)
        current_frame_kpts = torch.zeros(self.config['training']['batch_size'],
                                         self.config['model']['n_feature_maps'], 3)

        first_frame_gmaps = self.kpts2gmaps(first_frame_kpts)
        current_frame_gmaps = self.kpts2gmaps(current_frame_kpts)

        combined_representation = torch.cat([current_frame_gmaps, first_frame_fmaps, first_frame_gmaps], dim=1)
        combined_representation = add_coord_channels(combined_representation)

        np.testing.assert_array_equal(
            x=tuple(combined_representation.shape),
            y=(self.config['training']['batch_size'],
               2 * self.config['model']['n_feature_maps'] + self.enc_out_ch + 2,
               self.config['model']['feature_map_height'], self.config['model']['feature_map_width'])
        )

        rec_input = self.dec(combined_representation)
        np.testing.assert_array_equal(
            x=tuple(rec_input.shape),
            y=(self.config['training']['batch_size'], 3, 64, 64)
        )


if __name__ == "__main__":
    unittest.main()

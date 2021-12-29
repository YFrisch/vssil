import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
import numpy as np

from src.models.utils import init_weights
from .layers import Conv2d, Conv2DSamePadding
from .ulosd_layers import FeatureMapsToCoordinates, FeatureMapsToKeyPoints,\
    KeyPointsToFeatureMaps, add_coord_channels
from .ulosd_encoders import make_encoder, make_appearance_encoder
from .ulosd_decoder import make_decoder


class ULOSD(nn.Module):

    def __init__(self,
                 input_shape: tuple,
                 config: dict):
        super(ULOSD, self).__init__()

        self.device = config['device']

        self.feature_map_width = config['model']['feature_map_width']
        self.feature_map_height = config['model']['feature_map_height']

        """ 
            Image encoder.

            Iteratively halving the input width and doubling
            the number of channels, until the width of the
            feature maps is reached.
        """
        self.encoder, encoder_input_shape = make_encoder(input_shape, config)
        self.encoder.apply(lambda model: init_weights(m=model, config=config))

        self.appearance_net = make_appearance_encoder(input_shape, config)
        self.appearance_net.apply(lambda model: init_weights(m=model, config=config))

        """
            Image decoder.

            Iteratively doubling the input width and halving
            the number of channels, until the width of the
            original input image is reached.
        """
        self.decoder = make_decoder(encoder_input_shape, config)
        self.decoder.apply(lambda model: init_weights(m=model, config=config))

        """
            Ops layers
                    
        """
        self.maps_2_key_points = FeatureMapsToKeyPoints()
        self.key_points_2_maps = KeyPointsToFeatureMaps(
            sigma=config['model']['feature_map_gauss_sigma'],
            heatmap_width=config['model']['feature_map_width']
        )

    def encode(self,
               image_sequence: torch.Tensor,
               appearance: bool = False,
               re_sample_kpts: bool = False,
               re_sample_scale: float = 0.1) -> (torch.Tensor, torch.Tensor):
        """ Encodes a series of images into a tuple of a series of feature maps and
            key-points.

        :param image_sequence: Tensor of image series in (N, T, C, H, W)
        :param appearance: Set true if the input is only the first frame.
                (Appearance network in the paper; Same as decoder but without final soft-plus layer)
        :param re_sample_kpts: If set to 'True', the extracted key-points are used to define gaussian functions,
                               that are used to re-sample the actual key-points. This introduces additional noise.
        :param re_sample_scale: Scale of gaussian re-sampling

        """
        N, T = image_sequence.shape[0:2]

        # Unstack time
        image_list = [image_sequence[:, t, ...] for t in range(T)]
        feature_map_list = []
        key_point_list = []

        # Encode each time-step separately
        for image in image_list:
            if appearance:
                feature_maps = self.appearance_net(image)
                key_points = torch.empty((1, 1))
            else:
                image = add_coord_channels(image, device=self.device)
                feature_maps = self.encoder(image)
                key_points = self.maps_2_key_points(feature_maps)
                if re_sample_kpts:
                    # Stack batch and key-point dims together
                    N, K, D = key_points.shape
                    extracted_key_points = key_points.view((N*K, D))
                    # Define gaussians over mean vector and std of 0.1, for height and width positions
                    gauss_h = Normal(loc=extracted_key_points[..., 0], scale=re_sample_scale)
                    gauss_w = Normal(loc=extracted_key_points[..., 1], scale=re_sample_scale)
                    # Re-sample key-points
                    kpt_h = gauss_h.rsample().view((N, K, 1)).clip(-1.0, 1.0)
                    kpt_w = gauss_w.rsample().view((N, K, 1)).clip(-1.0, 1.0)
                    key_points = torch.cat([kpt_h, kpt_w, extracted_key_points[..., 2:3].view((N, K, 1))], dim=-1)
            key_point_list.append(key_points)
            feature_map_list.append(feature_maps)

        # Stack time
        feature_maps = torch.stack(feature_map_list, dim=1).to(self.device)  # -> (N, T, K, H', W')
        key_points = torch.stack(key_point_list, dim=1).to(self.device)  # -> (N, T, K, 3)

        return feature_maps, key_points

    def decode(self,
               keypoint_sequence: torch.Tensor,
               first_frame: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """ Decodes a series of key-points into a series of images.

        :param first_frame: Image of first time-step in (N, 1, 3, H, W)
        :param keypoint_sequence: Key-point sequence in (N, T, C, 3)
        :return: Reconstructed images, Reconstructed gaussian maps
        """

        T, K = keypoint_sequence.shape[1:3]

        # Encode first frame
        first_frame_feature_maps, _ = self.encode(first_frame, appearance=True)
        first_frame_feature_maps = first_frame_feature_maps.squeeze(1)
        # first_frame_key_points = torch.clone(keypoint_sequence[:, 0, ...]).unsqueeze(1).to(self.device)
        # first_frame_reconstructed_maps = self.key_points_2_maps(first_frame_key_points.squeeze(1)).to(self.device)
        first_frame_gaussian_maps = self.key_points_2_maps(keypoint_sequence[:, 0, ...]).to(self.device)

        # Unstack time
        key_points_list = [keypoint_sequence[:, t, ...] for t in range(T)]
        image_list = []
        gmap_list = []

        for key_points in key_points_list:

            gaussian_maps = self.key_points_2_maps(key_points).to(self.device)
            gmap_list.append(gaussian_maps)
            assert gaussian_maps.ndim == 4

            # Concat representation of current gaussian map and the information from the first frame
            combi = torch.cat(
                [gaussian_maps, first_frame_gaussian_maps, first_frame_feature_maps],
                dim=1
            )

            # Extend channels
            combi = add_coord_channels(combi, device=self.device)

            # Decode
            reconstructed_image = self.decoder(combi).to(self.device)
            # TODO: Mapping to [0, 1] range
            reconstructed_image = torch.sigmoid(reconstructed_image)
            image_list.append(reconstructed_image)

        # Stack time
        reconstructed_images = torch.stack(image_list, dim=1).to(self.device)
        gaussian_maps = torch.stack(gmap_list, dim=1).to(self.device)

        assert reconstructed_images.ndim == 5
        return reconstructed_images, gaussian_maps

    def forward(self, image_sequence: torch.Tensor) -> torch.Tensor:

        # Encode series
        feature_map_series, key_point_series = self.encode(image_sequence, appearance=False)

        # Decode encodings
        reconstructed_images, _ = self.decode(
            keypoint_sequence=key_point_series,
            first_frame=image_sequence[:, 0, ...].unsqueeze(1)
        )

        return reconstructed_images


class ULOSD_Dist_Parallel(nn.parallel.DistributedDataParallel):

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(ULOSD_Dist_Parallel, self).__init__(module, device_ids=device_ids, output_device=output_device, dim=dim)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def load_state_dict(self, state_dict, strict=True):
        super(ULOSD_Dist_Parallel, self).load_state_dict(state_dict, strict)

    def encode(self, image_sequence: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        return self.module.encode(image_sequence)

    def decode(self, keypoint_sequence: torch.Tensor, first_frame: torch.Tensor) -> torch.Tensor:
        return self.module.decode(keypoint_sequence, first_frame)

    def init_weights(self, m: torch.nn.Module):
        return self.module.init_weights(m)


class ULOSD_Parallel(nn.DataParallel):

    def __init__(self, module, device_ids=None, output_device=None, dim=0):
        super(ULOSD_Parallel, self).__init__(module, device_ids=device_ids, output_device=output_device, dim=dim)

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)

    def load_state_dict(self, state_dict, strict=True):
        super(ULOSD_Parallel, self).load_state_dict(state_dict, strict)

    def encode(self, image_sequence: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        return self.module.encode(image_sequence)

    def decode(self, keypoint_sequence: torch.Tensor, first_frame: torch.Tensor) -> torch.Tensor:
        return self.module.decode(keypoint_sequence, first_frame)

    def init_weights(self, m: torch.nn.Module):
        return self.module.init_weights(m)

    def forward(self, image_sequence: torch.Tensor) -> torch.Tensor:
        return self.module(image_sequence)



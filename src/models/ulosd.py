import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

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

        self.conv_weight_init = config['model']['conv_init']

        """ 
            Image encoder.

            Iteratively halving the input width and doubling
            the number of channels, until the width of the
            feature maps is reached.
        """
        self.encoder, encoder_input_shape = make_encoder(input_shape, config)
        self.encoder.apply(self.init_weights)

        self.appearance_net = make_appearance_encoder(input_shape, config)
        self.appearance_net.apply(self.init_weights)

        """
            Image decoder.

            Iteratively doubling the input width and halving
            the number of channels, until the width of the
            original input image is reached.
        """
        self.decoder = make_decoder(encoder_input_shape, config)
        self.decoder.apply(self.init_weights)

        """
            Ops layers
                    
        """
        self.maps_2_key_points = FeatureMapsToKeyPoints()
        self.key_points_2_maps = KeyPointsToFeatureMaps(
            sigma=1.0,
            heatmap_width=config['model']['feature_map_width']
        )

    def init_weights(self, m: torch.nn.Module):
        if type(m) == nn.Conv2d:
            if self.conv_weight_init == 'he_uniform':
                torch.nn.init.kaiming_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            if self.conv_weight_init == 'xavier_uniform':
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            if self.conv_weight_init == 'ones':
                torch.nn.init.ones_(m.weight)
                m.bias.data.fill_(1.00)

    def encode(self, image_sequence: torch.Tensor, appearance: bool = False) -> (torch.Tensor, torch.Tensor):
        """ Encodes a series of images into a tuple of a series of feature maps and
            key-points.

        :param image_sequence: Tensor of image series in (N, T, C, H, W)
        :param appearance: Set true if the input is only the first frame.
                (Appearance network in the paper; Same as decoder but without
                 final softplus layer)
        """
        # Flatten (N, T, C, H, W) into (N*T, C, H, W)
        N = image_sequence.shape[0]
        T = image_sequence.shape[1]

        # Unstack time
        image_list = [image_sequence[:, t, ...] for t in range(T)]
        maps_list = []
        key_points_list = []
        # x = image_sequence.view((N*T, *image_sequence.shape[2:]))

        for image in image_list:
            if appearance:
                feature_maps = self.appearance_net(image)
                key_points = torch.empty((1, 1))
            else:
                feature_maps = self.encoder(image)
                key_points = self.maps_2_key_points(feature_maps)
            key_points_list.append(key_points)
            maps_list.append(feature_maps)

        # Unstack time
        feature_maps = torch.stack(maps_list, dim=1).to(self.device)
        key_points = torch.stack(key_points_list, dim=1).to(self.device)

        return feature_maps, key_points

    def decode(self,
               keypoint_sequence: torch.Tensor,
               first_frame: torch.Tensor) -> torch.Tensor:
        """ Decodes a series of key-points into a series of images.

        :param first_frame: Image of first time-step in (N, 1, 3, H, W)
        :param keypoint_sequence: Key-point sequence in (N, T, C, 3)
        """

        T = keypoint_sequence.shape[1]
        C = keypoint_sequence.shape[2]
        key_points_shape = (T, C, 3)

        # Encode first frame
        first_frame_feature_maps, _ = self.encode(first_frame, appearance=True)
        first_frame_feature_maps = first_frame_feature_maps.squeeze(1).to(self.device)
        first_frame_key_points = torch.clone(keypoint_sequence[:, 0, ...]).unsqueeze(1).to(self.device)
        first_frame_reconstructed_maps = self.key_points_2_maps(first_frame_key_points.squeeze(1)).to(self.device)

        key_points_list = [keypoint_sequence[:, t, ...] for t in range(T)]
        image_list = []

        for key_points in key_points_list:

            gaussian_maps = self.key_points_2_maps(key_points).to(self.device)
            assert gaussian_maps.ndim == 4

            # Concat representation of current gaussian map and the information from the first frame
            combi = torch.cat(
                [gaussian_maps, first_frame_reconstructed_maps, first_frame_feature_maps.squeeze(1)],
                dim=1
            )

            # Extend channels
            combi = add_coord_channels(combi, device=self.device)

            # Decode
            reconstructed_image = self.decoder(combi).to(self.device)
            image_list.append(reconstructed_image)

        # Stack time-steps
        reconstructed_images = torch.stack(image_list, dim=1).to(self.device)

        assert reconstructed_images.ndim == 5
        return reconstructed_images

    def forward(self, image_sequence: torch.Tensor) -> torch.Tensor:

        # Encode series
        feature_map_series, key_point_series = self.encode(image_sequence, appearance=False)

        # Decode encodings
        reconstructed_images = self.decode(
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



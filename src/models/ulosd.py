import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from .layers import Conv2d, Conv2DSamePadding
from .ulosd_layers import FeatureMapsToCoordinates, FeatureMapsToKeyPoints,\
    KeyPointsToFeatureMaps, add_coord_channels


class ULOSD(nn.Module):

    def __init__(self,
                 input_shape: tuple,
                 config: dict):
        super(ULOSD, self).__init__()

        self.feature_map_width = config['model']['feature_map_width']
        self.feature_map_height = config['model']['feature_map_height']

        """
            Image encoder.
            
            Iteratively halving the input width and doubling
            the number of channels, until the width of the
            feature maps is reached.
        """
        self.encoder = []
        # Adjusted input shape (for add_coord_channels)
        encoder_input_shape = (input_shape[1] + 2, *input_shape[2:])
        print(encoder_input_shape)
        assert len(encoder_input_shape) == 3

        # First, expand the input to an initial number of filters
        # num_channels = encoder_input_shape[0]
        num_channels = 3
        self.encoder.append(
            Conv2DSamePadding(
                in_channels=num_channels,
                out_channels=config['model']['n_init_filters'],
                kernel_size=(config['model']['conv_kernel_size'], config['model']['conv_kernel_size']),
                stride=(1, 1),
                activation=nn.Identity()
            )
        )
        input_width = encoder_input_shape[-1]
        num_channels = config['model']['n_init_filters']
        # Apply additional layers
        for _ in range(config['model']['n_convolutions_per_res']):
            self.encoder.append(
                Conv2DSamePadding(
                    in_channels=num_channels,
                    out_channels=num_channels,
                    kernel_size=(config['model']['conv_kernel_size'], config['model']['conv_kernel_size']),
                    stride=(1, 1),
                    activation=nn.Identity()
                )
            )

        while True:
            # Reduce resolution
            self.encoder.append(
                Conv2DSamePadding(
                    in_channels=num_channels,
                    out_channels=num_channels*2,
                    kernel_size=(config['model']['conv_kernel_size'], config['model']['conv_kernel_size']),
                    stride=(2, 2),
                    activation=nn.LeakyReLU(negative_slope=0.2)
                )
            )
            # Apply additional layers
            for _ in range(config['model']['n_convolutions_per_res']):
                self.encoder.append(
                    Conv2DSamePadding(
                        in_channels=num_channels * 2,
                        out_channels=num_channels * 2,
                        kernel_size=(config['model']['conv_kernel_size'], config['model']['conv_kernel_size']),
                        stride=(1, 1),
                        activation=nn.LeakyReLU(negative_slope=0.2)
                    )
                )
            input_width //= 2
            num_channels *= 2
            if input_width <= config['model']['feature_map_width']:
                break

        # Final layer that maps to the desired number of feature_maps
        self.encoder.append(
            Conv2DSamePadding(
                in_channels=num_channels,
                out_channels=config['model']['n_feature_maps'],
                kernel_size=(config['model']['conv_kernel_size'], config['model']['conv_kernel_size']),
                stride=(1, 1),
                activation=nn.Softplus()
            )
        )
        self.encoder = nn.Sequential(*self.encoder)

        """
            Image decoder.

            Iteratively doubling the input width and halving
            the number of channels, until the width of the
            original input image is reached.
        """
        self.decoder = []
        num_channels = config['model']['n_feature_maps'] * 3 + 2
        # num_levels = np.log2(init_input_width / config['model']['feature_map_width'])
        num_levels = np.log2(encoder_input_shape[1] / config['model']['feature_map_width'])
        if num_levels % 1:
            raise ValueError(f"The input image width must be a two potency"
                             f" of the feature map width, but got {encoder_input_shape[1]}"
                             f" and {config['model']['feature_map_width']}!")

        # Iteratively double the resolution by upsampling
        for _ in range(int(num_levels)):

            num_out_channels = num_channels//2

            self.decoder.append(
                nn.Upsample(
                    scale_factor=(2.0, 2.0),
                    mode='bilinear',
                    align_corners=True
                )
            )

            self.decoder.append(
                Conv2DSamePadding(
                    in_channels=num_channels,
                    out_channels=num_out_channels,
                    kernel_size=(config['model']['conv_kernel_size'], config['model']['conv_kernel_size']),
                    stride=(1, 1),
                    activation=nn.ReLU()
                )
            )

            for _ in range(config['model']['n_convolutions_per_res'] - 1):
                self.decoder.append(
                    Conv2DSamePadding(
                        in_channels=num_out_channels,
                        out_channels=num_out_channels,
                        kernel_size=(config['model']['conv_kernel_size'], config['model']['conv_kernel_size']),
                        stride=(1, 1),
                        activation=nn.ReLU()
                    )
                )

            num_channels //= 2

        # Adjust channels
        self.decoder.append(
            Conv2DSamePadding(
                in_channels=num_out_channels,
                out_channels=3,
                kernel_size=(1, 1),
                stride=(1, 1),
                activation=nn.Identity()
            )
        )

        self.decoder = nn.Sequential(*self.decoder)

        """
            Ops layers
                    
        """
        self.maps_2_key_points = FeatureMapsToKeyPoints()
        self.key_points_2_maps = KeyPointsToFeatureMaps(
            sigma=1.0,
            heatmap_width=config['model']['feature_map_width']
        )

    def encode(self, image_sequence: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        """ Encodes a series of images into a tuple of a series of feature maps and
            key-points.

        :param image_sequence: Tensor of image series in (N, T, C, H, W)
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
            feature_maps = self.encoder(image)
            maps_list.append(feature_maps)

            key_points = self.maps_2_key_points(feature_maps)
            key_points_list.append(key_points)

        # Unstack time
        feature_maps = torch.stack(maps_list, dim=1)
        key_points = torch.stack(key_points_list, dim=1)

        return feature_maps, key_points

    def decode(self,
               keypoint_sequence: torch.Tensor,
               first_frame: torch.Tensor) -> torch.Tensor:
        """ Decodes a series of key-points into a series of images.

        :param first_frame: Image of first time-step in (N, 1, 3, H, W)
        :param keypoint_sequence: Key-point sequence in (N, T, C, 3)
        """

        # TODO num_timesteps =
        T = keypoint_sequence.shape[1]
        C = keypoint_sequence.shape[2]
        key_points_shape = (T, C, 3)

        # Encode first frame
        first_frame_feature_maps, first_frame_key_points = self.encode(first_frame)
        first_frame_reconstructed_maps = self.key_points_2_maps(first_frame_key_points.squeeze(1))

        key_points_list = [keypoint_sequence[:, t, ...] for t in range(T)]
        image_list = []

        for key_points in key_points_list:

            gaussian_maps = self.key_points_2_maps(key_points)
            assert gaussian_maps.ndim == 4

            # Concat representation of current gaussian map and the information from the first frame
            combi = torch.cat(
                [gaussian_maps, first_frame_feature_maps.squeeze(1), first_frame_reconstructed_maps],
                dim=1
            )

            # Extend channels
            combi = add_coord_channels(combi)

            # Decode
            reconstructed_image = self.decoder(combi)
            image_list.append(reconstructed_image)

        # Stack time-steps
        reconstructed_images = torch.stack(image_list, dim=1)

        # TODO: Add first frame as in the google code

        assert reconstructed_images.ndim == 5
        return reconstructed_images

    def forward(self, image_sequence: torch.Tensor) -> torch.Tensor:

        # Encode series
        feature_map_series, key_point_series = self.encode(image_sequence)

        # Decode encodings
        reconstructed_images = self.decode(
            keypoint_sequence=key_point_series,
            first_frame=image_sequence[:, 0, ...].unsqueeze(1)
        )

        return reconstructed_images



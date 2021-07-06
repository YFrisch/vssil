import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from .layers import Conv2d, Conv2DSamePadding
from .ulosd_layers import FeatureMapsToCoordinates, FeatureMapsToKeyPoints, KeyPointsToFeatureMaps

class ULOSD(nn.Module):

    def __init__(self,
                 input_width: int,
                 input_height: int,
                 config: dict):
        super(ULOSD, self).__init__()

        init_input_width = input_width
        self.feature_map_width = config['model']['feature_map_width']
        self.feature_map_height = config['model']['feature_map_height']

        """
            Image encoder.
            
            Iteratively halving the input width and doubling
            the number of channels, until the width of the
            feature maps is reached.
        """
        self.encoder = []
        num_channels = 3

        # First, expand the input to an initial number of filters
        self.encoder.append(
            Conv2DSamePadding(
                in_channels=num_channels,
                out_channels=config['model']['n_init_filters'],
                kernel_size=(config['model']['conv_kernel_size'], config['model']['conv_kernel_size']),
                stride=(1, 1),
                activation=nn.Identity()
            )
        )
        num_channels = config['model']['n_init_filters']
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

        while input_width > config['model']['feature_map_width']:
            self.encoder.append(
                Conv2DSamePadding(
                    in_channels=num_channels,
                    out_channels=num_channels*2,
                    kernel_size=(config['model']['conv_kernel_size'], config['model']['conv_kernel_size']),
                    stride=(2, 2),
                    activation=nn.LeakyReLU(negative_slope=0.2)
                )
            )
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
        self.encoder.append(
            Conv2DSamePadding(
                in_channels=num_channels,
                out_channels=config['model']['n_feature_maps'],
                kernel_size=(config['model']['conv_kernel_size'], config['model']['conv_kernel_size']),
                stride=(2, 2),
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
        num_channels = config['model']['n_feature_maps']
        # num_levels = np.log2(init_input_width / config['model']['feature_map_width'])
        num_levels = int(init_input_width / config['model']['feature_map_width'])
        if num_levels % 1:
            raise ValueError(f"The input image width must be an integral multiple"
                             f" of the feature map width, but got {init_input_width}"
                             f" and {config['model']['feature_map_width']}!")

        for _ in range(num_levels):

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

        self.decoder = nn.Sequential(*self.decoder)

        """
            Ops layers
                    
        """
        self.maps_2_key_points = FeatureMapsToKeyPoints()
        self.key_points_2_maps = KeyPointsToFeatureMaps(
            sigma=1.0,
            heatmap_width=config['model']['feature_map_width']
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """ Encode given image, aka return keypoint coordinates. """
        feature_maps = self.encoder(x)
        normalized_maps = self.normalize_feature_maps(feature_maps)
        feature_positions = self.average_feature_maps(normalized_maps)
        return feature_positions

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """ Decode given latent representation."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Flatten (N, T, C, H, W) into (N*T, C, H, W)
        N = x.shape[0]
        T = x.shape[1]
        x = x.view((N*T, *x.shape[2:]))

        print(f"Input after flattening: {tuple(x.shape)}")

        # Calc. "raw" features
        R = self.encoder(x)
        print(f"Feature maps: ", R.shape)

        # Make encodings ((x, y, mu) triples)
        latent = self.maps_2_key_points(R)
        print(f"Latent: {tuple(latent.shape)}")

        # Make reconstructed feature maps
        recons_maps = self.key_points_2_maps(latent)
        print("Reconstructed maps: ", recons_maps.shape)

        # Decode feature maps
        reconstructed_images = self.decoder(recons_maps)
        print(f"Reconstructed images: {tuple(reconstructed_images.shape)}")

        reconstr_images = reconstructed_images.view((N, T, *reconstructed_images.shape[1:]))

        return reconstructed_images



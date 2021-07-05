import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from .layers import Conv2d


class ULOSD(nn.Module):

    def __init__(self,
                 input_width: int,
                 input_height: int,
                 config: dict):
        super(ULOSD, self).__init__()

        init_input_width = input_width

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
            Conv2d(
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
                Conv2d(
                    in_channels=num_channels,
                    out_channels=num_channels,
                    kernel_size=(config['model']['conv_kernel_size'], config['model']['conv_kernel_size']),
                    stride=(1, 1),
                    activation=nn.Identity()
                )
            )

        while input_width > config['model']['feature_map_width']:
            input_width //= 2
            self.encoder.append(
                Conv2d(
                    in_channels=num_channels,
                    out_channels=num_channels*2,
                    kernel_size=(config['model']['conv_kernel_size'], config['model']['conv_kernel_size']),
                    stride=(2, 2),
                    activation=nn.LeakyReLU(negative_slope=0.2)
                )
            )
            for _ in range(config['model']['n_convolutions_per_res']):
                self.encoder.append(
                    Conv2d(
                        in_channels=num_channels * 2,
                        out_channels=num_channels * 2,
                        kernel_size=(config['model']['conv_kernel_size'], config['model']['conv_kernel_size']),
                        stride=(1, 1),
                        activation=nn.LeakyReLU(negative_slope=0.2)
                    )
                )
            num_channels *= 2
        self.encoder.append(
            Conv2d(
                in_channels=num_channels,
                out_channels=config['model']['n_feature_maps'],
                kernel_size=(config['model']['conv_kernel_size'], config['model']['conv_kernel_size']),
                stride=(2, 2),
                activation=nn.Softplus()
            )
        )
        input_width //= 2
        self.encoder = nn.Sequential(*self.encoder)
        print(self.encoder)

        """
            Image decoder.

            Iteratively doubling the input width and halving
            the number of channels, until the width of the
            original input image is reached.
        """
        self.decoder = []
        while input_width < init_input_width:

            self.decoder.append(
                Conv2d(
                    in_channels=num_channels,
                    out_channels=num_channels // 2,
                    kernel_size=(config['model']['conv_kernel_size'], config['model']['conv_kernel_size']),
                    stride=(2, 2),
                    activation=nn.ReLU()
                )
            )

            for _ in range(config['model']['n_convolutions_per_res']):
                self.decoder.append(
                    Conv2d(
                        in_channels=num_channels // 2,
                        out_channels=num_channels // 2,
                        kernel_size=(config['model']['conv_kernel_size'], config['model']['conv_kernel_size']),
                        stride=(1, 1),
                        activation=nn.ReLU()
                    )
                )

            input_width *= 2
            num_channels //= 2

        self.decoder = nn.Sequential(*self.decoder)

    def calc_conv_channels(self, config: dict, input_width: int, input_height: int) -> tuple:
        """ Calculates the required parameters for the convolutional layers. """
        map_w = config['model']['feature_map_width']
        map_h = config['model']['feature_map_height']
        map_c = config['model']['n_feature_maps']
        n_convs = config['model']['n_convolutions']
        channel_list = []
        return ()

    def normalize_feature_maps(self, maps: torch.Tensor) -> torch.Tensor:
        """ Normalizes a given feature map.

        :param maps: Feature maps in (H, W, k)
        :return: Normalized feature maps in (H, W, k)
        """
        for k in range(0, maps.shape[1]):
            pixel_sum = torch.sum(maps[:, k, ...], dim=[1, 2], keepdim=False)
            for b in range(maps.shape[0]):
                maps[b, ...] /= pixel_sum[b]
        return maps

    def keypoint_presence(self, maps: torch.Tensor) -> torch.Tensor:
        """ Calculates μ for every key-point.

        :param maps: Feature maps in (B, K, H, W)
        :return: Presence values in (B, K, 1)
        """
        B = maps.shape[0]
        K = maps.shape[1]
        val = torch.empty((B, K, 1))
        for b in range(B):
            # Iterate over maps
            for k in range(K):
                h = maps[b, k, ...].shape[0]
                w = maps[b, k, ...].shape[1]
                sum = torch.sum(maps[b, k, ...], dim=(0, 1))
                val[b, k, :] = sum/(h*w)
        return val

    def average_feature_maps(self, maps: torch.Tensor) -> torch.Tensor:
        """ Averages a given feature map to a single (x, y) coordinate.

        :param maps: Normalized feature maps in (H, W, k)
        :return: ((x, y),k) feature point coordinates
        """
        batch_size = maps.shape[0]
        n_feature_maps = maps.shape[1]
        coords = torch.zeros(size=(batch_size, n_feature_maps, 2))
        for k in range(n_feature_maps):
            for v in range(maps[:, k, ...].shape[1]):
                for u in range(maps[:, k, ...].shape[2]):
                    for b in range(maps.shape[0]):
                        coords[b, k, ...] += torch.tensor([v, u])*maps[b, k, v, u]
        return coords

    def make_gaussian_blob_maps(self, x: torch.Tensor) -> torch.Tensor:
        """ Makes K maps containing a 'Gaussian blob'
            with a given s.d. around the key-point location

        :param x: Latent representation in (N, K, 3)
        :return: Feature map in (N, K, H, W)
        """
        N = x.shape[0]
        K = x.shape[1]
        height = 16
        width = 16
        maps = torch.empty((N, K, height, width))
        sigma = 1
        for n in range(N):
            for k in range(K):
                for u in range(0, height):
                    for v in range(0, width):
                        maps[n, k, u, v] = x[n, k, 2] * torch.exp(
                            torch.tensor([
                                -(1 / 2 * sigma ** 2) * torch.norm(torch.tensor([u, v]) - x[n, k, :2])**2
                            ]))
        return maps

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

        print(x.shape)
        x = F.pad(x, )

        # Calc. "raw" features
        R = self.encoder(x)

        # Calc. key-point presence
        mu = self.keypoint_presence(R)

        # Normalize features
        D = self.normalize_feature_maps(R)

        # Average to obtain coordinates
        latent = self.average_feature_maps(D)

        # Make encodings ((x, y, mu) triples)
        latent = torch.cat([latent, mu], dim=2)
        print(latent.shape)

        # Recreate gaussian feature maps from latent
        maps = self.make_gaussian_blob_maps(latent)
        print(maps.shape)

        # Decode feature maps
        reconstructed_images = self.decoder(maps)
        print(reconstructed_images.shape)

        reconstr_images = reconstructed_images.view((N, T, *reconstructed_images.shape[1:]))

        return reconstructed_images



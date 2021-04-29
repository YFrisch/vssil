""" Implements the Deep Spatial Autoencoder as used in https://arxiv.org/pdf/1509.06113.pdf

    It's architecture mainly consists of
    - Encoder consisting of 3 conv. layers with ReLU activations
    - Spatial Soft Arg-Max Operation
    - Fully Connected Decoder from low-dim. encoding to downscaled greyscale reconstruction of input image
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import FullyConnected, BatchNormConv2D, SpatialSoftArgmax


class DeepSpatialAE(nn.Module):
    def __init__(self, config: dict = None):

        assert config is not None, "No configuration given for DeepSpatialAE"

        super().__init__()

        assert config['conv']['out_channels']*2 == config['fc']['in_features'],\
            "Latent dimension mismatch between Encoder and Decoder!"

        self.conv1 = BatchNormConv2D(in_channels=config['conv']['in_channels'],
                                     out_channels=config['conv']['hidden_sizes'][0],
                                     activation=nn.ReLU(),
                                     kernel_size=(7, 7),
                                     stride=(2, 2))

        self.conv2 = BatchNormConv2D(in_channels=config['conv']['hidden_sizes'][0],
                                     out_channels=config['conv']['hidden_sizes'][1],
                                     activation=nn.ReLU(),
                                     kernel_size=(5, 5))

        self.conv3 = BatchNormConv2D(in_channels=config['conv']['hidden_sizes'][1],
                                     out_channels=config['conv']['out_channels'],
                                     activation=nn.ReLU(),
                                     kernel_size=(5, 5))

        self.spatial_soft_argmax = SpatialSoftArgmax(temperature=config['spatial']['temperature'],
                                                     normalize=config['spatial']['normalize'])

        self.decoder = FullyConnected(in_features=config['fc']['in_features'],
                                      out_features=(config['fc']['out_img_height'] *
                                                    config['fc']['out_img_width']),
                                      activation=nn.Sigmoid())

        self.out_img_shape = (1, config['fc']['out_img_height'], config['fc']['out_img_width'])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """ Returns the features/latent variables for given image (batch),
            i.e. encodes the image.

        :param x: Input image in (N, C, H, W) format
        """

        _y = self.conv1(x)
        _y = self.conv2(_y)
        _y = self.conv3(_y)

        _y = self.spatial_soft_argmax(_y)

        return _y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the whole architecture."""
        n, c, h, w = x.size()

        # Encoder:
        _y = self.encode(x)
        n, c, _2 = _y.size()

        # Decoder:
        _y = self.decoder(_y.view(n, c*2)).view(n, *self.out_img_shape)

        return _y

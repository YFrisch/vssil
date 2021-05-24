""" Multi-frame Time Contrastive Network , as used in
    https://arxiv.org/pdf/1808.00928.pdf
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import Conv2d, Conv3d, FullyConnected


class MultiFrameTCN(nn.Module):

    def __init__(self,
                 n_frames: int = 5,
                 in_dims: (tuple, torch.Size) = None,
                 n_convolutions: int = 3,
                 channels: tuple = (3, 256, 128, 64),
                 channels_3d: int = 16,
                 conv_act_func: nn.Module = nn.ReLU(),
                 n_fc_layers: int = 3,
                 fc_layer_dims=(256, 256, 32),
                 fc_act_func: (nn.Module, list) = nn.ReLU(),
                 last_act_func: nn.Module = nn.Identity(),
                 ):
        """ Creates class instance.

        :param n_frames: Number of frames to stack for loss calculation
        :param n_convolutions: Depth of conv. layer
        :param channels: Channels of convolutional layers
        :param channels_3d: Output channels of 3D convolution / spatial averaging
        :param conv_act_func: Activation function(s) for convolutional layers
        :param n_fc_layers: Number of fully connected layers after spatial averaging
        :param fc_layer_dims: Sizes of fully connected layers
        :param fc_act_func: Activation function(s) for fc layers
        :param last_act_func: Activation function for output layer
        """

        assert n_convolutions == len(channels) - 1
        assert n_fc_layers == len(fc_layer_dims)

        super(MultiFrameTCN, self).__init__()

        # 2D convolution, applied to each frame
        self.conv_2d = nn.Sequential(
            *[Conv2d(
                in_channels=channels[i],
                out_channels=channels[i+1],
                activation=conv_act_func
            ) for i in range(0, n_convolutions)]
        )

        # 3D convolution, applied to stacked outputs of 2D conv.
        # (Temporal Aggregation)
        self.conv_3d = Conv3d(in_channels=channels[-1],
                              out_channels=channels_3d,
                              activation=conv_act_func)

        self.flatten = nn.Flatten()
        flatten_size = self.calc_flatten_size(in_dims, n_frames)[1]

        # Fully connected network for dimensionality reduction
        self.fc = nn.Sequential(
            FullyConnected(
                in_features=flatten_size,
                out_features=fc_layer_dims[0],
                activation=fc_act_func
            ),
            *[FullyConnected(
                in_features=fc_layer_dims[i],
                out_features=fc_layer_dims[i+1],
                activation=fc_act_func
            ) for i in range(0, n_fc_layers - 2)],
            FullyConnected(
                in_features=fc_layer_dims[-2],
                out_features=fc_layer_dims[-1],
                activation=last_act_func
            )
        )

    def calc_flatten_size(self, in_dims: (tuple, torch.Size), n_frames: int) -> (tuple, torch.Size):
        """ Calculates the size of the flattened output of the 3D convolution.

        :param in_dims: Dimensions of single model input (N, T, C, H, W)
        :param n_frames: Frames to stack per model forward pass
        :return: Shape of flattened 3D conv. output
        """
        n, t, c, h, w = in_dims

        # Fake input of size (n x t), c , h , w
        # x = torch.stack([torch.ones(size=in_dims) for _ in range(0, n_frames)], dim=0)
        x = torch.ones(size=(n, t, c, h, w))
        _y = self.convolute(x)
        return _y.shape

    def convolute(self, x: torch.Tensor) -> torch.Tensor:
        """ Performs convolution + flatten steps on input tensor."""
        n, t, c, h, w = x.shape

        _y = self.conv_2d(x.view(n*t, c, h, w))

        _, c, h, w = _y.shape
        _y = _y.view((n, t, c, h, w))

        _y = self.conv_3d(_y.permute(0, 2, 1, 3, 4))  # Conv3D requires (N, C, T, H, W))

        _y = self.flatten(_y)

        return _y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the architecture.
            NOTE: The input is in (N, T, C, H, W) and
            every frame (T (fixed)) should be forwarded through
            the conv. layer.

        :param x: Model input
        :return: Model output
        """
        _y = self.convolute(x)
        _y = self.fc(_y)
        return _y

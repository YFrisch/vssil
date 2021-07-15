import torch
import torch.nn as nn

from .layers import Conv2d, SpatialSoftArgmax, FullyConnected
from .inception3 import CustomInception3


class TimeContrastiveNetwork(nn.Module):

    def __init__(self,
                 n_convolutions: int = 2,
                 conv_channels: tuple = (),
                 embedding_size: int = 32):
        """ Creates class instance.

        :param n_convolutions: Number of conv. layers after inception net.

        """
        super(TimeContrastiveNetwork, self).__init__()

        assert len(conv_channels) == n_convolutions+1

        self.inception_net = CustomInception3()

        self.conv = nn.Sequential(
            *[Conv2d(
                in_channels=conv_channels[i],
                out_channels=conv_channels[i+1],
                activation=nn.ReLU()
            ) for i in range(0, n_convolutions)]
        )
        self.spatial_softmax = SpatialSoftArgmax(
            temperature=0.9,
            normalize=False
        )

        self.flatten = nn.Flatten()
        self.flatten_size = conv_channels[-1] * 2

        self.fully_connected = FullyConnected(
            in_features=self.flatten_size,
            out_features=embedding_size,
            activation=nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the architecture.

        :param x: Input tensor
        :return: Output tensor / prediction (embedding)
        """
        _y = self.inception_net(x)
        _y = self.conv(_y)
        _y = self.spatial_softmax(_y)
        _y = self.flatten(_y)
        _y = self.fully_connected(_y)
        return _y

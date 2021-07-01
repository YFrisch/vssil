import torch
import torch.nn as nn

from .layers import Conv2d

class ULOSD(nn.Module):

    def __init__(self,
                 config: dict):
        super(ULOSD, self).__init__()

        self.encoder = nn.Sequential(
            *[Conv2d(
                in_channels=...,
                out_channels=...,
                stride=(2, 2)
            ) for _ in range(config['model']['n_convolutions'])]
        )
        self.decoder = ...

    def average_feature_map(self, map_k: torch.Tensor) -> torch.Tensor:
        """ Averages a given feature map to a single (x, y) coordinate.

        :param map_k: Feature map in (H, W, 1)
        :return: (x, y) coordinates
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
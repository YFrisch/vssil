import torch
import torch.nn as nn

from .layers import Conv2d

class ULOSD(nn.Module):

    def __init__(self,
                 config: dict):
        super(ULOSD, self).__init__()

        conv_channels = self.calc_conv_channels(config)

        self.encoder = nn.Sequential(
            *[Conv2d(
                in_channels=conv_channels[i],
                out_channels=conv_channels[i+1],
                stride=(2, 2),
                activation=nn.ReLU
            ) for i in range(config['model']['n_convolutions'] - 1)],
            Conv2d(
                in_channels=conv_channels[-1],
                out_channels=config['model']['n_feature_maps'],
                stride=(2, 2),
                activation=nn.Softplus
            )
        )
        self.decoder = ...

    def calc_conv_channels(self, config: dict) -> tuple:
        """ Calculates the required parameters for the convolutional layers. """
        map_w = config['model']['feature_map_width']
        map_h = config['model']['feature_map_height']
        map_c = config['model']['n_feature_maps']
        return ()

    def normalize_feature_map(self, map_k: torch.Tensor) -> torch.Tensor:
        """ Normalizes a given feature map.

        :param map_k: Feature map in (H, W, 1)
        :return: Normalized feature map in (H, W, 1)
        """
        pixel_sum = torch.sum(map_k, dim=[0, 1], keepdim=False)
        return map_k/pixel_sum

    def average_feature_map(self, map_k: torch.Tensor) -> torch.Tensor:
        """ Averages a given feature map to a single (x, y) coordinate.

        :param map_k: Feature map in (H, W, 1)
        :return: (x, y) coordinates
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass
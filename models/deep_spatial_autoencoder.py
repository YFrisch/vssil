import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import FullyConnected, BatchNormConv2D, SpatialSoftmax


class DeepSpatialAE(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.conv1 = BatchNormConv2D(in_channels=in_channels, out_chanels=64, activation=nn.ReLU,
                                     kernel_size=(7, 7), stride=(2, 2))
        self.conv2 = BatchNormConv2D(in_channels=64, out_chanels=32, activation=nn.ReLU,
                                     kernel_size=(5, 5))
        self.conv3 = BatchNormConv2D(in_channels=32, out_chanels=16, activation=nn.ReLU,
                                     kernel_size=(5, 5))
        self.spatial_softmax = SpatialSoftmax(in_channels=16)
        self.decoder = FullyConnected(in_features=32, out_features=60*60, activation=nn.Sigmoid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _y = self.conv1(x)
        _y = self.conv2(_y)
        _y = self.conv3(_y)
        print(f"Post conv.: {_y.shape}")
        _y = self.spatial_softmax(_y)
        print(f"Pre fc: {_y.shape}")
        _y = self.decoder(_y)
        print(f"Reconstruced Img: {_y.shape}")
        return _y

import torch
import torch.nn as nn


class FullyConnected(nn.Module):

    def __init__(self, in_features: int, out_features: int, activation=None):
        super().__init__()
        self.linear = nn.Linear(in_features=in_features, out_features=out_features)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _y = self.linear(x)
        if self.activation is not None:
            _y = self.activation(_y, inplace=True)
        return _y


class BatchNormConv2D(nn.Module):

    def __init__(self, in_channels: int, out_chanels: int, activation=None, **args):
        super().__init__()
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_chanels, **args)
        self.batch_norm = nn.BatchNorm2d(num_features=out_chanels, eps=1e-3)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _y = self.conv2d(x)
        _y = self.batch_norm(_y)
        if self.activation is not None:
            _y = self.activation(_y, inplace=True)
        return _y


class SpatialSoftmax(nn.Module):

    def __init__(self, in_channels: int):
        super().__init__()
        self.spatial_softmax = nn.Softmax2d()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _y = self.spatial_softmax(x)

        # Max over channels
        _y = _y.max(dim=1)
        return _y

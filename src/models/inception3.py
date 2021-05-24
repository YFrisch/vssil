import torch
import torch.nn as nn
from torchvision.models.inception import BasicConv2d, InceptionA


class CustomInception3(nn.Module):
    """ This is part of the Inception 3 model from 
        https://github.com/pytorch/vision/blob/master/torchvision/models/inception.py
    """
    
    def __init__(self):
        super(CustomInception3, self).__init__()

        self.conv2d_1a_3x3 = BasicConv2d(in_channels=3,
                                         out_channels=32,
                                         kernel_size=3,
                                         stride=2)
        self.conv2d_2a_3x3 = BasicConv2d(in_channels=32,
                                         out_channels=32,
                                         kernel_size=3)
        self.conv2d_2b_3x3 = BasicConv2d(in_channels=32,
                                         out_channels=64,
                                         kernel_size=3,
                                         padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2d_3b_1x1 = BasicConv2d(in_channels=64,
                                         out_channels=80,
                                         kernel_size=1)
        self.conv2d_4a_3x3 = BasicConv2d(in_channels=80,
                                         out_channels=192,
                                         kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.mixed_5b = InceptionA(in_channels=192, pool_features=32)
        self.mixed_5c = InceptionA(in_channels=256, pool_features=64)
        self.mixed_5d = InceptionA(in_channels=288, pool_features=64)

        self.out_shape = (288, 35, 35)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # N x 3 x 299 x 299
        x = self.conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.mixed_5d(x)
        # N x 288 x 35 x 35
        return x

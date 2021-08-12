import torch
import torch.nn as nn

from .utils import activation_dict


class TransporterBlock(nn.Module):

    """ Torch module 'block' consisting of a
        convolutional layer, a batch normalization layer,
        and an activation function.
    """

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple,
                 stride: tuple,
                 padding: tuple,
                 activation: str,
                 skip_connections: bool = False):
        super(TransporterBlock, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        self.batch_norm = nn.BatchNorm2d(out_channels)

        self.activation = activation_dict[activation]()

        self.skip_connections = skip_connections

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _x = self.conv(x)
        _x = self.batch_norm(_x)
        return self.activation(_x)


class TransporterEncoder(nn.Module):

    def __init__(self,
                 config: dict):
        super(TransporterEncoder, self).__init__()

        self.net = nn.Sequential(
            TransporterBlock(in_channels=config['model']['num_img_channels'], out_channels=32,
                             kernel_size=(7, 7), stride=(1,), padding=(3,),
                             activation=config['model']['activation'],
                             skip_connections=config['model']['skip_connections']),
            TransporterBlock(in_channels=32, out_channels=32,
                             kernel_size=(3, 3), stride=(1,), padding=(1,),
                             activation=config['model']['activation'],
                             skip_connections=config['model']['skip_connections']),
            TransporterBlock(in_channels=32, out_channels=64,
                             kernel_size=(3, 3), stride=(2,), padding=(1,),
                             activation=config['model']['activation'],
                             skip_connections=config['model']['skip_connections']),
            TransporterBlock(in_channels=64, out_channels=64,
                             kernel_size=(3, 3), stride=(1,), padding=(1,),
                             activation=config['model']['activation'],
                             skip_connections=config['model']['skip_connections']),
            TransporterBlock(in_channels=64, out_channels=128,
                             kernel_size=(3, 3), stride=(2,), padding=(1,),
                             activation=config['model']['activation'],
                             skip_connections=config['model']['skip_connections']),
            TransporterBlock(in_channels=128, out_channels=128,
                             kernel_size=(3, 3), stride=(1,), padding=(1,),
                             activation=config['model']['activation'],
                             skip_connections=config['model']['skip_connections']),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

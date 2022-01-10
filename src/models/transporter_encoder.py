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

        if activation in ['LeakyRELU', 'LeakyReLU', 'LeakyRelu']:
            self.activation = activation_dict[activation](negative_slope=0.2)
        else:
            self.activation = activation_dict[activation]()

        self.skip_connections = skip_connections

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _x = self.conv(x)
        _x = self.batch_norm(_x)
        return self.activation(_x)


class TransporterEncoder(nn.Module):

    def __init__(self, config: dict):

        """ Creates class instance.

            Encoder consists of a series of the Transporter blocks above,
            until the desired hidden dimension size is reached.
        """

        super(TransporterEncoder, self).__init__()

        transporter_blocks = []
        ch = 32
        assert config['model']['hidden_dim'] % 32 == 0, "Choose hidden dim as multiple of 32"
        transporter_blocks.append(TransporterBlock(in_channels=config['model']['num_img_channels'], out_channels=32,
                                                   kernel_size=(7, 7), stride=(1,), padding=(3,),
                                                   activation=config['model']['activation'],
                                                   skip_connections=config['model']['skip_connections']))
        transporter_blocks.append(TransporterBlock(in_channels=32, out_channels=32,
                                                   kernel_size=(3, 3), stride=(1,), padding=(1,),
                                                   activation=config['model']['activation'],
                                                   skip_connections=config['model']['skip_connections']))
        while ch < config['model']['hidden_dim']:
            transporter_blocks.append(TransporterBlock(in_channels=ch, out_channels=ch*2,
                                                       kernel_size=(3, 3), stride=(2,), padding=(1,),
                                                       activation=config['model']['activation'],
                                                       skip_connections=config['model']['skip_connections']))
            transporter_blocks.append(TransporterBlock(in_channels=ch*2, out_channels=ch*2,
                                                       kernel_size=(3, 3), stride=(1,), padding=(1,),
                                                       activation=config['model']['activation'],
                                                       skip_connections=config['model']['skip_connections']))
            ch *= 2

        assert ch == config['model']['hidden_dim']
        self.net = nn.Sequential(*transporter_blocks)

    """
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
        # Addition:
        TransporterBlock(in_channels=128, out_channels=256,
                         kernel_size=(3, 3), stride=(2,), padding=(1,),
                         activation=config['model']['activation'],
                         skip_connections=config['model']['skip_connections']),
        TransporterBlock(in_channels=256, out_channels=256,
                         kernel_size=(3, 3), stride=(1,), padding=(1,),
                         activation=config['model']['activation'],
                         skip_connections=config['model']['skip_connections']),
        TransporterBlock(in_channels=256, out_channels=512,
                         kernel_size=(3, 3), stride=(2,), padding=(1,),
                         activation=config['model']['activation'],
                         skip_connections=config['model']['skip_connections']),
        TransporterBlock(in_channels=512, out_channels=512,
                         kernel_size=(3, 3), stride=(1,), padding=(1,),
                         activation=config['model']['activation'],
                         skip_connections=config['model']['skip_connections']),
        
    )
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

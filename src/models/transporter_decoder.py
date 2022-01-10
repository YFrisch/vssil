import torch
import torch.nn as nn

from .transporter_encoder import TransporterBlock


class TransporterDecoder(nn.Module):

    def __init__(self, config: dict):
        """ Creates class instance.

            Decoder consists of a series of Transporter blocks,
            consisting of stride 1 convolutions and 2D up-sampling,
            until the original input channel dimension is reached.
        """
        super(TransporterDecoder, self).__init__()

        transporter_blocks = []
        ch = config['model']['hidden_dim']
        assert config['model']['hidden_dim'] % 32 == 0, "Choose hidden dim as multiple of 32"
        while ch > 32:
            transporter_blocks.append(TransporterBlock(in_channels=ch, out_channels=ch,
                                                       kernel_size=(3, 3), stride=(1,), padding=(1,),
                                                       activation=config['model']['activation'],
                                                       skip_connections=config['model']['skip_connections']))
            transporter_blocks.append(TransporterBlock(in_channels=ch, out_channels=int(ch*0.5),
                                                       kernel_size=(3, 3), stride=(1,), padding=(1,),
                                                       activation=config['model']['activation'],
                                                       skip_connections=config['model']['skip_connections']))
            transporter_blocks.append(nn.UpsamplingBilinear2d(scale_factor=2))
            ch = int(ch*0.5)
        assert ch == 32
        transporter_blocks.append(TransporterBlock(in_channels=32, out_channels=32,
                                                   kernel_size=(3, 3), stride=(1,), padding=(1,),
                                                   activation=config['model']['activation'],
                                                   skip_connections=config['model']['skip_connections']))
        transporter_blocks.append(TransporterBlock(in_channels=32, out_channels=config['model']['num_img_channels'],
                                                   kernel_size=(7, 7), stride=(1,), padding=(3,),
                                                   activation='identity'))

        self.net = nn.Sequential(*transporter_blocks)

        """
        self.net = nn.Sequential(
            # Addition:
            TransporterBlock(in_channels=512, out_channels=512,
                             kernel_size=(3, 3), stride=(1,), padding=(1,),
                             activation=config['model']['activation'],
                             skip_connections=config['model']['skip_connections']),
            TransporterBlock(in_channels=512, out_channels=256,
                             kernel_size=(3, 3), stride=(1,), padding=(1,),
                             activation=config['model']['activation'],
                             skip_connections=config['model']['skip_connections']),
            nn.UpsamplingBilinear2d(scale_factor=2),
            TransporterBlock(in_channels=256, out_channels=256,
                             kernel_size=(3, 3), stride=(1,), padding=(1,),
                             activation=config['model']['activation'],
                             skip_connections=config['model']['skip_connections']),
            TransporterBlock(in_channels=256, out_channels=128,
                             kernel_size=(3, 3), stride=(1,), padding=(1,),
                             activation=config['model']['activation'],
                             skip_connections=config['model']['skip_connections']),
            nn.UpsamplingBilinear2d(scale_factor=2),


            TransporterBlock(in_channels=128, out_channels=128,
                             kernel_size=(3, 3), stride=(1,), padding=(1,),
                             activation=config['model']['activation'],
                             skip_connections=config['model']['skip_connections']),
            TransporterBlock(in_channels=128, out_channels=64,
                             kernel_size=(3, 3), stride=(1,), padding=(1,),
                             activation=config['model']['activation'],
                             skip_connections=config['model']['skip_connections']),
            nn.UpsamplingBilinear2d(scale_factor=2),
            TransporterBlock(in_channels=64, out_channels=64,
                             kernel_size=(3, 3), stride=(1,), padding=(1,),
                             activation=config['model']['activation'],
                             skip_connections=config['model']['skip_connections']),
            TransporterBlock(in_channels=64, out_channels=32,
                             kernel_size=(3, 3), stride=(1,), padding=(1,),
                             activation=config['model']['activation'],
                             skip_connections=config['model']['skip_connections']),
            nn.UpsamplingBilinear2d(scale_factor=2),
            TransporterBlock(in_channels=32, out_channels=32,
                             kernel_size=(3, 3), stride=(1,), padding=(1,),
                             activation=config['model']['activation'],
                             skip_connections=config['model']['skip_connections']),
            TransporterBlock(in_channels=32, out_channels=config['model']['num_img_channels'],
                             kernel_size=(7, 7), stride=(1,), padding=(3,),
                             activation='identity')
        )
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

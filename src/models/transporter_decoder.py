import torch
import torch.nn as nn

from .transporter_encoder import TransporterBlock


class TransporterDecoder(nn.Module):

    def __init__(self,
                 config: dict):
        super(TransporterDecoder, self).__init__()

        self.net = nn.Sequential(
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



import torch
import torch.nn as nn

from .transporter_encoder import TransporterBlock


class TransporterKeypointer(nn.Module):

    def __init__(self,
                 config: dict):
        super(TransporterKeypointer, self).__init__()
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
        self.regressor = nn.Conv2d(
            in_channels=128, out_channels=config['model']['num_keypoints'], kernel_size=(1, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.net(x)
        return self.regressor(x)


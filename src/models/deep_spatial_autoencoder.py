""" Implements the Deep Spatial Autoencoder as used in https://arxiv.org/pdf/1509.06113.pdf

    It's architecture mainly consists of
    - Encoder consisting of 3 conv. layers with ReLU activations
    - Spatial Soft Arg-Max Operation
    - Fully Connected Decoder from low-dim. encoding to downscaled greyscale reconstruction of input image
"""
import os.path

import torch
import torch.nn as nn

from src.models.utils import init_weights
from .layers import FullyConnected, BatchNormConv2D, SpatialSoftArgmax
from .utils import activation_dict


class DeepSpatialAE(nn.Module):
    def __init__(self, config: dict = None, device: str = 'cpu'):

        assert config is not None, "No configuration given for DeepSpatialAE"

        super().__init__()

        assert config['conv']['out_channels'] * 2 == config['fc']['in_features'], \
            "Latent dimension mismatch between Encoder and Decoder!"

        self.conv1 = BatchNormConv2D(in_channels=config['conv']['in_channels'],
                                     out_channels=config['conv']['hidden_sizes'][0],
                                     activation=activation_dict[config['conv']['activation']](),
                                     # activation=nn.ReLU(),
                                     kernel_size=(7, 7),
                                     padding=(3, 3),
                                     stride=(2, 2)
                                     )

        self.conv1 = self.imagenet_init(self.conv1)

        self.conv2 = BatchNormConv2D(in_channels=config['conv']['hidden_sizes'][0],
                                     out_channels=config['conv']['hidden_sizes'][1],
                                     activation=activation_dict[config['conv']['activation']](),
                                     # activation=nn.ReLU(),
                                     kernel_size=(5, 5),
                                     padding=(3, 3),
                                     stride=(1, 1)
                                     )

        self.conv3 = BatchNormConv2D(in_channels=config['conv']['hidden_sizes'][1],
                                     out_channels=config['conv']['out_channels'],
                                     activation=activation_dict[config['conv']['activation']](),
                                     # activation=nn.ReLU(),
                                     kernel_size=(5, 5),
                                     padding=(3, 3),
                                     stride=(1, 1)
                                     )

        self.spatial_soft_argmax = SpatialSoftArgmax(temperature=config['spatial']['temperature'],
                                                     normalize=config['spatial']['normalize'],
                                                     device=device)

        self.decoder = FullyConnected(in_features=config['fc']['in_features'],
                                      out_features=(config['fc']['out_img_height'] *
                                                    config['fc']['out_img_width']),
                                      # activation=nn.Sigmoid())
                                      activation=activation_dict[config['fc']['activation']]()
                                      )

        self.out_img_shape = (1, config['fc']['out_img_height'], config['fc']['out_img_width'])

    def imagenet_init(self, layer: torch.nn.Module) -> torch.nn.Module:
        if not os.path.isfile('src/models/pretrained/googlenet_imagenet.PTH'):
            os.makedirs('src/models/pretrained/', exist_ok=True)
            m = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
            torch.save(m, 'src/models/pretrained/googlenet_imagenet.PTH')
        else:
            m = torch.load('src/models/pretrained/googlenet_imagenet.PTH')

        layer.conv2d.weight.data = m.conv1.conv.weight.data
        layer.batch_norm.weight.data = m.conv1.bn.weight.data
        return layer

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """ Returns the features/latent variables for given image (batch),
            i.e. encodes the image.

        :param x: Input image in (N, C, H, W) or (N, T, C, H, W) format
        """
        reshape = False
        if x.dim() == 5:
            n, t, c, h, w = x.size()
            x = x.view((n * t, c, h, w))
            reshape = True

        _y = self.conv1(x)
        _y = self.conv2(_y)
        fmaps = self.conv3(_y)

        kpts = self.spatial_soft_argmax(fmaps)
        n_t, c, _2 = kpts.size()
        if reshape:
            kpts = kpts.view((n, t, c, _2))

        return kpts, fmaps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward pass through the whole architecture."""
        n, t, c, h, w = x.size()

        x = x.view((n * t, c, h, w))

        # Encoder:
        kpts, fmaps = self.encode(x)
        n_t, c, _2 = kpts.size()

        # Decoder:
        rec = self.decoder(kpts.view(n_t, c * 2)).view(n, t, *self.out_img_shape)

        return rec

    def train(self, mode: bool = True):
        return super(DeepSpatialAE, self).train(mode)

    def eval(self):
        return super(DeepSpatialAE, self).eval()

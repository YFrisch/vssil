import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from .vq_ema import VectorQuantizerEMA
from .ulosd_layers_modified import FeatureMapsToKeyPoints, KeyPointsToFeatureMaps


class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens // 2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens // 2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)

        x = self._conv_2(x)
        x = F.relu(x)

        x = self._conv_3(x)

        return self._residual_stack(x)


class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)

        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens,
                                                out_channels=num_hiddens // 2,
                                                kernel_size=4,
                                                stride=2, padding=1)

        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens // 2,
                                                out_channels=3,
                                                kernel_size=4,
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)

        x = self._residual_stack(x)

        x = self._conv_trans_1(x)
        x = F.relu(x)

        return self._conv_trans_2(x)


class KeyPointsToGaussianMaps(nn.Module):

    def __init__(self,
                 batch_size: int = 1,
                 time_steps: int = 1,
                 n_kpts: int = 1,
                 heatmap_width: int = 32,
                 device: str = 'cpu'):
        super(KeyPointsToGaussianMaps, self).__init__()
        self.device = device
        self.heatmap_width = heatmap_width
        x_range = torch.linspace(0, heatmap_width, heatmap_width)
        y_range = torch.linspace(0, heatmap_width, heatmap_width)
        self.x_range, self.y_range = torch.meshgrid(x_range, y_range)
        self.x_range = self.x_range.view((1, 1, heatmap_width, heatmap_width))
        self.x_range = self.x_range.expand(batch_size * time_steps, n_kpts, -1, -1)
        self.y_range = self.y_range.view((1, 1, heatmap_width, heatmap_width))
        self.y_range = self.y_range.expand(batch_size * time_steps, n_kpts, -1, -1)
        self.x_range = self.x_range.to(device)
        self.y_range = self.y_range.to(device)
        self.blank_map = torch.zeros((batch_size * time_steps, n_kpts, heatmap_width, heatmap_width)).to(device)

    def get_grid(self):
        return self.x_range, self.y_range

    def gaussian_2d_pdf(self,
                        x: torch.Tensor, y: torch.Tensor,
                        mean_x: torch.Tensor, mean_y: torch.Tensor,
                        sd_x: torch.Tensor, sd_y: torch.Tensor):
        # Expand mean to (N*T, K, H', W')
        mean_x = mean_x.view(*mean_x.shape, 1, 1)
        mean_x = mean_x.expand(-1, -1, self.heatmap_width, self.heatmap_width)

        mean_y = mean_y.view(*mean_y.shape, 1, 1)
        mean_y = mean_y.expand(-1, -1, self.heatmap_width, self.heatmap_width)

        # Expand sd to (N*T, K, H', W')
        sd_x_exp = (2 * torch.pow(sd_x, 2)).view(*x.shape[:2], 1, 1)
        sd_x_exp = sd_x_exp.expand(-1, -1, self.heatmap_width, self.heatmap_width)
        sd_y_exp = (2 * torch.pow(sd_x, 2)).view(*y.shape[:2], 1, 1)
        sd_y_exp = sd_y_exp.expand(-1, -1, self.heatmap_width, self.heatmap_width)

        denominator = 1 / (2 * math.pi * sd_x * sd_y)
        denominator_exp = denominator.view(*denominator.shape, 1, 1)
        denominator_exp = denominator_exp.expand(-1, -1, self.heatmap_width, self.heatmap_width)

        x_diff = torch.pow((x - mean_x), 2)
        y_diff = torch.pow((y - mean_y), 2)

        numerator = torch.exp(-(x_diff / sd_x_exp +
                                y_diff / sd_y_exp))

        return denominator_exp * numerator

    def forward(self, kpts: torch.Tensor) -> torch.Tensor:
        """ Converts a (N, T, K, 3) tensor of key-point coordinates
            into an (N, T, C, H', W') tensor of gaussian feature maps
            with key-point position as mean.
        """

        feature_map = self.blank_map + self.gaussian_2d_pdf(
            x=self.x_range, y=self.y_range,
            mean_x=kpts[..., 0], mean_y=kpts[..., 1],
            sd_x=kpts[..., 3], sd_y=kpts[..., 3])

        return feature_map


class VQ_VAE(nn.Module):

    def __init__(self):
        super(VQ_VAE, self).__init__()

        self._encoder = Encoder(in_channels=3,
                                num_hiddens=128,
                                num_residual_layers=2,
                                num_residual_hiddens=32)

        self._pre_vq_conv = nn.Conv2d(in_channels=128,
                                      out_channels=64,
                                      kernel_size=1,
                                      stride=1)

        self._vq_vae = VectorQuantizerEMA(num_embeddings=512,
                                          embedding_dim=64,
                                          commitment_cost=0.25,
                                          decay=0.99)

        self._decoder = Decoder(in_channels=64,
                                num_hiddens=128,
                                num_residual_layers=2,
                                num_residual_hiddens=32)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        rec = self._decoder(quantized)

        return loss, rec, perplexity


class VQ_VAE_KPT(nn.Module):

    def __init__(self,
                 batch_size: int = 1,
                 time_steps: int = 1,
                 num_embeddings: int = 1,
                 heatmap_width: int = 32,
                 encoder_in_channels: int = 3,
                 num_hiddens: int = 128,
                 embedding_dim: int = 5,
                 num_residual_layers: int = 2,
                 num_residual_hiddens: int = 32,
                 device: str = 'cpu'
                 ):
        super(VQ_VAE_KPT, self).__init__()

        # Shape specifications
        self.N, self.T = batch_size, time_steps
        self.C, self.H, self.W, self.Cp, self.Hp, self.Wp, self.K, self.D = \
            None, None, None, None, None, None, None, None

        self._encoder = Encoder(in_channels=encoder_in_channels,
                                num_hiddens=num_hiddens,
                                num_residual_layers=num_residual_layers,
                                num_residual_hiddens=num_residual_hiddens).to(device)

        self._appearance_encoder = Encoder(in_channels=encoder_in_channels,
                                           num_hiddens=embedding_dim,
                                           num_residual_layers=num_residual_layers,
                                           num_residual_hiddens=num_residual_hiddens).to(device)

        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens,
                                      out_channels=embedding_dim,
                                      kernel_size=1,
                                      stride=1).to(device)

        self._vq_vae = VectorQuantizerEMA(num_embeddings=num_embeddings,  # num_embeddings
                                          embedding_dim=embedding_dim,
                                          commitment_cost=0.25,
                                          decay=0.99).to(device)

        self._fmap2kpt = FeatureMapsToKeyPoints(device=device)

        self._kpt2gmap = KeyPointsToFeatureMaps(heatmap_width=heatmap_width, device=device)

        self._decoder = Decoder(in_channels=embedding_dim * 3,
                                # in_channels=n_kpts,  # embedding dim
                                num_hiddens=num_hiddens,
                                num_residual_layers=num_residual_layers,
                                num_residual_hiddens=num_residual_hiddens).to(device)

        self._gmap_decoder = Decoder(in_channels=embedding_dim,
                                     # in_channels=n_kpts,  # embedding dim
                                     num_hiddens=num_hiddens,
                                     num_residual_layers=num_residual_layers,
                                     num_residual_hiddens=num_residual_hiddens).to(device)

        self.heatmap_width = heatmap_width

    def transport(self,
                  source_gaussian_maps: torch.Tensor,
                  target_gaussian_maps: torch.Tensor,
                  source_feature_maps: torch.Tensor,
                  target_feature_maps: torch.Tensor):
        """ Transports features by suppressing features from the source image
            and adding features from the target image around its key-points.

        :param source_gaussian_maps:
        :param target_gaussian_maps:
        :param source_feature_maps:
        :param target_feature_maps:
        :return:
        """

        _out = source_feature_maps
        for s, t in zip(torch.unbind(source_gaussian_maps, 1), torch.unbind(target_gaussian_maps, 1)):
            _out = (1 - s.unsqueeze(1)) * (1 - t.unsqueeze(1)) * _out + t.unsqueeze(1) * target_feature_maps
        return _out

    def encode(self, image_sequence: torch.Tensor, verbose: bool = False) \
            -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):

        self.N, self.T, self.C, self.H, self.W = image_sequence.shape

        if verbose:
            print('Input: ', image_sequence.shape)

        fmaps = self._encoder(image_sequence.view((self.N * self.T, self.C, self.H, self.W)))

        if verbose:
            print('Encoded: ', fmaps.shape)

        fmaps = self._pre_vq_conv(fmaps)
        _, self.Cp, self.Hp, self.Wp = fmaps.shape

        if verbose:
            print('Pre VQ: ', fmaps.shape)

        vq_loss, quantized, perplexity, _ = self._vq_vae(fmaps)

        if verbose:
            print('Quantized: ', quantized.shape)

        # kpts = self._fmap2kpt(torch.flatten(z, start_dim=2))
        sp = F.softplus(quantized)
        kpts = self._fmap2kpt(sp)
        _, self.K, self.D = kpts.shape

        # Unstack time-steps into separate dimension
        kpts = kpts.view((self.N, self.T, self.K, self.D))
        fmaps = fmaps.view((self.N, self.T, self.Cp, self.Hp, self.Wp))

        if verbose:
            print('KeyPoints: ', kpts.shape)

        return quantized, fmaps, kpts, vq_loss, perplexity

    def decode_quantized_stream(self, quantized_maps_series: torch.Tensor, verbose: bool = False) -> torch.Tensor:

        rec = self._decoder(quantized_maps_series)\

        # Unstack time-steps into separate dimension
        rec = rec.view((self.N, self.T, self.C, self.H, self.W))

        if verbose:
            print('Reconstruction: ', rec.shape)

        return rec

    def decode_kpt_stream(self, key_point_series: torch.Tensor, verbose: bool = False) \
            -> (torch.Tensor, torch.Tensor):

        gmaps = self._kpt2gmap(key_point_series.view((self.N * self.T, self.K, self.D)))
        if verbose:
            print('Gaussian maps: ', gmaps.shape)

        gmap_rec = self._gmap_decoder(gmaps)

        # Unstack time-steps into separate dimension
        gmaps = gmaps.view((self.N, self.T, self.Cp, self.Hp, self.Wp))
        gmap_rec = gmap_rec.view((self.N, self.T, self.C, self.H, self.W))

        if verbose:
            print('Gaussian map reconstruction: ', gmap_rec.shape)

        return gmap_rec, gmaps

    def forward(self, image_sequence: torch.Tensor, verbose: bool = False) -> torch.Tensor:

        # Encode image series
        quantized_series, feature_map_series, key_point_series, _, _ = self.encode(image_sequence)

        # Decode quantized encodings
        reconstructed_images = self.decode(quantized_series)

        return reconstructed_images



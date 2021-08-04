import torch
import torch.nn as nn
import torch.nn.functional as F

from .transporter_encoder import TransporterEncoder
from .transporter_decoder import TransporterDecoder
from .transporter_keypointer import TransporterKeypointer


class Transporter(nn.Module):

    def __init__(self,
                 config: dict):
        super(Transporter, self).__init__()

        self.device = config['device']

        self.gaussian_map_std = config['model']['gaussian_map_std']
        self.conv_weight_init = config['model']['weight_init']

        self.encoder = TransporterEncoder(config)
        self.encoder.apply(self._init_weights)
        self.decoder = TransporterDecoder(config)
        self.decoder.apply(self._init_weights)
        self.keypointer = TransporterKeypointer(config)
        self.keypointer.apply(self._init_weights)

        # for n, p in self.encoder.named_parameters():
        #    print(f'name: {n}\t req. grad: {p.requires_grad}')
        # exit()

    def _keypoint_loc_mean(self, features: torch.Tensor) -> torch.Tensor:
        """

        :param features: Feature maps in (N, K, H, W)
        :return: Weighted average of feature maps in (N, K, 2)
        """

        # Normalized weight for each column/row across each row/column
        row_weight = F.softmax(features.mean(-1), dim=-1)
        column_weight = F.softmax(features.mean(-2), dim=-1)

        row_u = row_weight.mul(torch.linspace(-1, 1, row_weight.size(-1),
                                              dtype=features.dtype,
                                              device=features.device)).sum(-1)

        column_u = column_weight.mul(torch.linspace(-1, 1,
                                                    column_weight.size(-1),
                                                    dtype=features.dtype,
                                                    device=features.device)).sum(-1)

        return torch.stack((row_u, column_u), -1)

    def _gaussian_map(self, features: torch.Tensor) -> torch.Tensor:
        width, height = features.shape[-1], features.shape[-2]
        mean = self._keypoint_loc_mean(features)
        mean_y, mean_x = mean[..., 0:1], mean[..., 1:2]
        y = torch.linspace(-1.0, 1.0, height, dtype=mean.dtype, device=mean.device)
        x = torch.linspace(-1.0, 1.0, width, dtype=mean.dtype, device=mean.device)
        mean_y, mean_x = mean_y.unsqueeze(-1), mean_x.unsqueeze(-1)

        y = torch.reshape(y, [1, 1, height, 1])
        x = torch.reshape(x, [1, 1, 1, width])

        inv_std = 1.0 / torch.FloatTensor([self.gaussian_map_std])
        inv_std = inv_std.to(self.device)
        g_y = torch.pow(y - mean_y, 2).to(self.device)
        g_x = torch.pow(x - mean_x, 2).to(self.device)
        dist = (g_y + g_x) * inv_std ** 2
        g_yx = torch.exp(-dist)
        # g_yx = g_yx.permute([0, 2, 3, 1])
        return g_yx

    def _transport(self,
                   source_keypoints: torch.Tensor,
                   target_keypoints: torch.Tensor,
                   source_features: torch.Tensor,
                   target_features: torch.Tensor):
        _out = source_features
        for s, t in zip(torch.unbind(source_keypoints, 1), torch.unbind(target_keypoints, 1)):
            _out = (1 - s.unsqueeze(1)) * (1 - t.unsqueeze(1)) * _out + t.unsqueeze(1) * target_features
        return _out

    def _spatial_softmax(self, features: torch.Tensor) -> torch.Tensor:
        """ Computes the softmax over the spatial dimensions
            Compute the softmax over height and width

        :param features: Tensor of shape [N, C, H, W]
        :return: Spatial softmax of input tensor
        """
        features_reshape = features.reshape(features.shape[:-2] + (-1,))
        output = F.softmax(features_reshape, dim=-1)
        output = output.reshape(features.shape)
        return output

    def _init_weights(self, m: torch.nn.Module):
        if type(m) == nn.Conv2d:
            if self.conv_weight_init == 'he_uniform':
                torch.nn.init.kaiming_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            if self.conv_weight_init == 'xavier_uniform':
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
            if self.conv_weight_init == 'ones':
                torch.nn.init.ones_(m.weight)
                m.bias.data.fill_(1.00)

    def forward(self, source_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:
        source_features = self.encoder(source_img)
        target_features = self.encoder(target_img)

        source_keypoints = self._gaussian_map(self._spatial_softmax(self.keypointer(source_img)))
        target_keypoints = self._gaussian_map(self._spatial_softmax(self.keypointer(target_img)))

        transported_features = self._transport(source_keypoints.detach(),
                                               target_keypoints,
                                               source_features.detach(),
                                               target_features)

        assert transported_features.shape == target_features.shape

        reconstruction = self.decoder(transported_features)

        return reconstruction

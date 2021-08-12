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
        self.spatial_softmax_layer = nn.Softmax2d()
        self.softmax_layer_width = nn.Softmax(dim=-1)
        self.softmax_layer_height = nn.Softmax(dim=-2)

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

        source_feature_maps = self.encoder(source_img)
        source_keypoints, source_gaussian_maps = self.keypointer(source_img)

        target_feature_maps = self.encoder(target_img)
        target_keypoints, target_gaussian_maps = self.keypointer(target_img)

        transported_features = self.transport(
            source_gaussian_maps.detach(),
            target_gaussian_maps,
            source_feature_maps.detach(),
            target_feature_maps)

        assert transported_features.shape == target_feature_maps.shape

        reconstruction = self.decoder(transported_features)

        return reconstruction

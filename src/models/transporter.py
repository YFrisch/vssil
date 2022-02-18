import torch
import torch.nn as nn

from src.models.utils import init_weights
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
        self.encoder.apply(lambda model: init_weights(m=model, config=config))
        self.decoder = TransporterDecoder(config)
        self.decoder.apply(lambda model: init_weights(m=model, config=config))
        self.keypointer = TransporterKeypointer(config)
        self.keypointer.apply(lambda model: init_weights(m=model, config=config))

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

    def forward(self, source_img: torch.Tensor, target_img: torch.Tensor) -> torch.Tensor:

        """

        :param source_img: Source image frame in (N, C, H, W)
        :param target_img: Target image frame in (N, C, H, W)
        :return:
        """

        with torch.no_grad():
            source_feature_maps = self.encoder(source_img)
            source_keypoints, source_gaussian_maps, source_kpt_maps = self.keypointer(source_img)

        target_feature_maps = self.encoder(target_img)
        target_keypoints, target_gaussian_maps, target_kpt_maps = self.keypointer(target_img)

        # TODO
        #source_penalty = torch.max(torch.sum(nn.functional.softplus(source_kpt_maps), dim=[-3]))
        #target_penalty = torch.max(torch.sum(nn.functional.softplus(target_kpt_maps), dim=[-3]))

        transported_features = self.transport(
            # source_gaussian_maps.detach(),
            source_gaussian_maps,
            target_gaussian_maps,
            # source_feature_maps.detach(),
            source_feature_maps,
            target_feature_maps)

        assert transported_features.shape == target_feature_maps.shape

        reconstruction = self.decoder(transported_features)
        """
        print(reconstruction.min())
        print(reconstruction.max())
        print(target_img.min())
        print(target_img.max())
        exit()
        """
        # TODO: Right act. function? Depends also on pre-processing
        # reconstruction = torch.sigmoid(reconstruction)
        # reconstruction = torch.tanh(reconstruction)

        return reconstruction  # , source_penalty, target_penalty

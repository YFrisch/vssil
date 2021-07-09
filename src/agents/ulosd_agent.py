import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.models.ulosd import ULOSD
from src.losses.temporal_separation_loss import temporal_separation_loss
from .abstract_agent import AbstractAgent


class ULOSD_Agent(AbstractAgent):

    def __init__(self,
                 dataset: Dataset,
                 config: dict
                 ):
        """ Creates class instance.

        :param dataset: Dataset to use for training and validation
        :param config: Dictionary of parameters
        """
        super(ULOSD_Agent, self).__init__(
            name="ULOSD Agent",
            dataset=dataset,
            config=config
        )

        N = config['training']['batch_size']
        T = config['model']['n_frames']
        C = 3
        # TODO: Make this modular
        H = 160
        W = 160
        input_shape = (T, C, H, W)

        self.model = ULOSD(
            input_shape=input_shape,
            config=config
        ).to(self.device)

        self.optim = torch.optim.Adam(
            params=self.model.parameters(),
            lr=config['training']['lr']
        )

    def preprocess(self, x: torch.Tensor, config: dict) -> (torch.Tensor, (torch.Tensor, torch.Tensor)):
        return x, x

    def loss_func(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """ Normalized L2 loss for image reconstruction

        :param prediction: Sequence of predicted images in (N, T, C, H, W)
        :param target: Actual image sequence in (N, T, C, H, W)
        :return: Normalized L2 loss between prediction and target
        """
        N, T = target.shape[0], target.shape[1]
        loss = F.mse_loss(input=prediction, target=target)
        loss /= (N*T)
        return loss

    def train_step(self, sample: torch.Tensor, target: torch.Tensor, config: dict) -> torch.Tensor:
        """ One step of training.

        :param sample: Image sequence in (N, T, C, H, W)
        :param target: -
        :param config: Configuration dictionary
        :return: TODO
        """

        sample, target = sample.to(self.device), target.to(self.device)

        # Vision model
        feature_maps, observed_key_points = self.model.encode(sample)
        reconstructed_images = self.model.decode(observed_key_points, sample[:, 0, ...].unsqueeze(1))

        # Dynamics model
        # TODO: Not used yet

        # Losses
        reconstruction_loss = self.loss_func(prediction=reconstructed_images,
                                             target=sample)

        separation_loss = temporal_separation_loss(cfg=config, coords=observed_key_points)

        # feature-map (L1) regularization of the activations of the last layer
        l1_penalty = config['model']['feature_map_regularization'] * torch.norm(feature_maps, p=1)

        # TODO: Not used yet
        coord_pred_loss = 0
        kl_loss = 0
        kl_scale = 0

        L = reconstruction_loss + separation_loss + coord_pred_loss + (kl_loss * kl_scale)


        L.backward()

        # Clip gradient norm
        nn.utils.clip_grad_norm_(self.model.parameters(), config['training']['clip_norm'])

        self.optim.step()

        return L

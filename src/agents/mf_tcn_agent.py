import torch
import torch.nn as nn
from torch.utils.data import Dataset
from ast import literal_eval

from src.models.mf_tcn import MultiFrameTCN
from .abstract_agent import AbstractAgent


class MF_TCN_Agent(AbstractAgent):

    """ Agent for Multi-frame Time Contrastive Networks."""

    def __init__(self,
                 dataset: Dataset,
                 config: dict
                 ):
        """ Creates class instance. """
        super(MF_TCN_Agent, self).__init__(
            name="Multi-frame TCN Agent",
            dataset=dataset,
            config=config
        )

        self.n_frames = config['model']['n_frames']

        self.model = MultiFrameTCN(
            n_frames=config['model']['n_frames'],
            in_dims=(config['training']['batch_size'], *dataset.__getitem__(0).shape),
            n_convolutions=config['model']['n_convolutions'],
            channels=literal_eval(config['model']['channels']),
            channels_3d=config['model']['channels_3d'],
            conv_act_func=nn.ReLU(),
            n_fc_layers=config['model']['n_fc_layers'],
            fc_layer_dims=literal_eval(config['model']['fc_layer_dims']),
            fc_act_func=nn.ReLU(),
            last_act_func=nn.Identity(),
        ).to(self.device)

    def preprocess(self, x: torch.Tensor, config: dict) -> (torch.Tensor, (torch.Tensor, torch.Tensor)):
        pass

    def loss_func(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pass

    def train_step(self, sample: torch.Tensor, target: torch.Tensor, config: dict) -> torch.Tensor:
        """ Performs one training step, returns loss.

        :param sample: Input tensor
        :param target: Target prediction tensor
        :param config: Config dict
        :return: Loss
        """
        self.optim.zero_grad()

        prediction = self.model(sample)

        loss = self.loss_func(prediction=prediction, target=target)

        loss.backward()

        self.optim.step()

        return loss







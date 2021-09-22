from ast import literal_eval

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.optim import Adam

from src.models.time_contrastive_network import TimeContrastiveNetwork
from src.models.utils import load_inception_weights
from .abstract_agent import AbstractAgent


class TCN_Agent(AbstractAgent):

    """ Agent for Multi-frame Time Contrastive Networks."""

    def __init__(self,
                 dataset: Dataset,
                 config: dict
                 ):
        """ Creates class instance. """
        super(TCN_Agent, self).__init__(
            name="Time Contrastive Network Agent",
            dataset=dataset,
            config=config
        )

        self.n_frames = config['model']['n_frames']
        self.n_views = config['model']['n_views']

        if self.n_views > 1:
            raise NotImplementedError("Multi-view TCN not yet supported.")

        self.model = TimeContrastiveNetwork(
            n_convolutions=config['model']['n_convolutions'],
            conv_channels=literal_eval(config['model']['conv_channels']),
            embedding_size=config['model']['embedding_size']
        ).to(self.device)

        self.optim = Adam(params=self.model.parameters(),
                          lr=config['training']['lr'])

        load_inception_weights(inception_net=self.model.inception_net, config=config)

    def preprocess(self, x: torch.Tensor, config: dict) -> (torch.Tensor, (torch.Tensor, torch.Tensor)):
        """ Create a triplet loss tuple from the sample.
            The middle frame is used as anchor, and positive and negative are sampled randomly
            from within / outside the positive range.

        :param x: Sample in (N, T, C, H, W)
        :param config: Configuration dictionary
        :return: (sample, target) tuple of torch tensors
        """

        n, t, c, h, w = x.size()
        _x = torch.empty(size=(n, 3, c, h, w))

        # Iterate over batches
        for n_i in range(0, n):
            # Use middle frame for anchoryyyyy
            anchor = int(config['model']['n_frames']/2)

            # Sample any index in positive range from anchor, for the positive example
            positive = anchor
            while positive == anchor:
                positive = np.random.randint(low=anchor-config['model']['positive_range'],
                                             high=anchor+config['model']['positive_range'])
            # Sample any index outside positive range from anchor, for the negative example
            negative = anchor
            while anchor-config['model']['positive_range'] <= negative <= anchor+config['model']['positive_range']:
                negative = np.random.randint(low=0, high=config['model']['n_frames'])

            # Define processed sample
            sample = torch.stack([x[n_i, anchor, ...],
                                  x[n_i, positive, ...],
                                  x[n_i, negative, ...]], dim=1)

            _x[n_i, ...] = sample


        # Format first two dims into one
        n, t, c, h, w = _x.size()
        _x = _x.view((n*t, c, h, w))

        return _x, torch.empty([])

    def loss_func(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """ Triplet loss.

            NOTE: prediction will be in (N*3, C)
        """
        _n = prediction.shape[0]
        n = int(_n/3)

        # (N, 3, C)
        _prediction = prediction.view((n, 3, *prediction.shape[1:]))

        # TODO: Is this the correct way for batch learning???
        loss_sum = 0
        for n_i in range(0, n):
            anchor_embedding = _prediction[n_i, 0, ...]
            positive_embedding = _prediction[n_i, 1, ...]
            negative_embedding = _prediction[n_i, 2, ...]

            pos = anchor_embedding@positive_embedding
            neg = anchor_embedding@negative_embedding

            loss = max(0, neg-pos)
            loss_sum = loss_sum + loss

        loss = torch.Tensor([loss_sum/n])
        loss.requires_grad = True

        return loss

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







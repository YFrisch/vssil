""" Agent implementation for the Deep Spatial Auto-Encoder as used in
    https://arxiv.org/pdf/1509.06113.pdf


    # TODO: Make sure, batch sizes are handled correctly, i.e. tensors in (N, T, C, H, W) format
"""
import torch
from torch.nn.functional import mse_loss, interpolate
from torchvision.transforms.functional import rgb_to_grayscale
import numpy as np
from tqdm import tqdm

from models.deep_spatial_autoencoder import DeepSpatialAE
from .abstract_agent import AbstractAgent


class SpatialAEAgent(AbstractAgent):

    def __init__(self, config: dict = None):
        super(SpatialAEAgent, self).__init__(name="SpatialAEAgent", config=config)
        self.smoothness_penalty = config['training']['smoothness_penalty']
        self.setup(config)

    def setup(self, config: dict = None):

        self.model = DeepSpatialAE(config)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=config['training']['lr'])

    def loss_func(self,
                  prediction: torch.Tensor,
                  target: torch.Tensor = None,
                  ft_minus1=None,
                  ft=None,
                  ft_plus1=None):

        """ Deep Spatial AE loss function, as in the paper.

        :param prediction: Reconstructed, greyscale image
        :param target: Target greyscale image
        :param ft_minus1: Features of previous image
        :param ft: Features of target image
        :param ft_plus1: Features of next image
        """

        penalty = torch.zeros(1, device=prediction.device)
        loss = mse_loss(input=prediction, target=target)

        if self.smoothness_penalty:
            penalty = mse_loss(ft_plus1 - ft, ft - ft_minus1)

        return loss + penalty

    def preprocess(self, x: dict, config: dict) -> (torch.Tensor, torch.Tensor):
        """ Returns sample and target from sampled data.

            In this case, the sampled image series is down-sampled and
            transformed to greyscale to give the target.
        """
        img_series = x['hd_kinect_img_series'][0]

        target = interpolate(img_series,
                             size=(config['fc']['out_img_width'],
                                   config['fc']['out_img_height']
                                   )
                             )
        target = rgb_to_grayscale(target)

        return img_series, target

    def train(self, config: dict = None):

        for epoch in range(config['training']['epochs']):

            self.model.train()

            self.optim.zero_grad()

            losses = []

            print("\n")

            for i, sample in enumerate(tqdm(self.train_data_loader)):

                sample, target = self.preprocess(sample, config)

                timesteps = sample.shape[0]

                for t in range(timesteps):

                    sample_t = sample[t].unsqueeze(0)
                    target_t = target[t].unsqueeze(0)

                    # Forward pass
                    prediction = self.model(sample_t)

                    assert prediction.shape == target_t.shape, \
                        f"Prediction shape {prediction.shape} does not match " \
                        f"target shape {target[t].unsqueeze(0).shape}"

                    # Loss
                    features_t_minus1 = self.model.encode(sample[t-1].unsqueeze(0)) if t > 0 else \
                        self.model.encode(sample_t)
                    features_t = self.model.encode(sample_t)
                    features_t_plus1 = self.model.encode(sample[t+1].unsqueeze(0)) if t < timesteps-1 else \
                        self.model.encode(sample_t)

                    loss = self.loss_func(prediction=prediction,
                                          target=target_t,
                                          ft_minus1=features_t_minus1,
                                          ft=features_t,
                                          ft_plus1=features_t_plus1)

                    losses.append(loss.detach().cpu().numpy())

                    loss.backward()

                    self.optim.step()

                    del loss, features_t_plus1, features_t, features_t_minus1

                del sample, target, prediction

                if i == 10:
                    break

            print(f"\nEpoch: {epoch}|{config['training']['epochs']}\t\t Avg. loss: {np.mean(losses)}")

    def validate(self, config: dict = None):
        raise NotImplementedError

    def evaluate(self, config: dict = None) -> (torch.Tensor, torch.Tensor):
        """ Evaluate trained agent on evaluation data.

            TODO: Add predictiveness?
        """

        self.model.eval()
        loss_per_sample = []

        with torch.no_grad():
            for i, sample in enumerate(tqdm(self.eval_data_loader)):

                sample, target = self.preprocess(sample, config)

                prediction = self.model(sample)

                sample_loss = mse_loss(prediction, target)
                loss_per_sample.append(sample_loss.cpu().numpy())

                if i == 10:
                    break

        print(f"##### Evaluation: Average loss: {np.mean(loss_per_sample)}")

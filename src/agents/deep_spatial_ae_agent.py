""" Agent implementation for the Deep Spatial Auto-Encoder as used in
    https://arxiv.org/pdf/1509.06113.pdf


    # TODO: Make sure, batch sizes are handled correctly, i.e. tensors in (N, T, C, H, W) format
"""
import time

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import mse_loss, interpolate
from torchvision.transforms.functional import rgb_to_grayscale
import numpy as np
from tqdm import tqdm

from .abstract_agent import AbstractAgent
from ..models.deep_spatial_autoencoder import DeepSpatialAE
from ..data.utils import play_video


class SpatialAEAgent(AbstractAgent):

    def __init__(self,
                 dataset: Dataset = None,
                 config: dict = None):

        super(SpatialAEAgent, self).__init__("Deep Spatial Auto-Encoder",
                                             dataset=dataset,
                                             config=config)

        self.smoothness_penalty = config['training']['smoothness_penalty']

    def setup(self, config: dict = None):

        print(f"\n##### Setting up {self.name} on {self.device}.")

        self.model = DeepSpatialAE(config).to(self.device)

        self.optim = torch.optim.Adam(self.model.parameters(),
                                      lr=config['training']['lr'],
                                      weight_decay=config['training']['weight_decay'])

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
        img_series = x['hd_kinect_img_series']

        target = interpolate(img_series,
                             size=(3,
                                   config['fc']['out_img_width'],
                                   config['fc']['out_img_height']
                                   )
                             )
        target = rgb_to_grayscale(target)

        return img_series, target

    def train(self, config: dict = None):

        print("\n##### Training:")

        for epoch in range(config['training']['epochs']):

            if not epoch % int(config['training']['epochs'])/5:
                self.make_train_val_split(config)

            self.model.train()

            self.optim.zero_grad()

            losses = []

            print("\n")
            time.sleep(1)

            for i, sample in enumerate(tqdm(self.train_data_loader)):

                with torch.no_grad():
                    sample, target = self.preprocess(sample, config)  # (N, T, C, H, W)

                sample, target = sample.to(self.device), target.to(self.device)

                assert sample.ndim == 5
                assert target.ndim == 5

                timesteps = sample.shape[1]

                for t in range(timesteps):

                    # TODO: For whole trajectories, the memory usage of this loop grow to much!

                    sample_t = sample[:, t, ...]
                    target_t = target[:, t, ...]

                    # Forward pass
                    prediction = self.model(sample_t)

                    assert prediction.shape == target_t.shape, \
                        f"Prediction shape {prediction.shape} does not match " \
                        f"target shape {target[t].unsqueeze(0).shape}"

                    # Loss
                    features_t_minus1 = self.model.encode(sample[:, t-1, ...]) if t > 0 else \
                        self.model.encode(sample_t)
                    features_t = self.model.encode(sample_t)
                    features_t_plus1 = self.model.encode(sample[:, t+1, ...]) if t < timesteps-1 else \
                        self.model.encode(sample_t)

                    loss = self.loss_func(prediction=prediction,
                                          target=target_t,
                                          ft_minus1=features_t_minus1,
                                          ft=features_t,
                                          ft_plus1=features_t_plus1)

                    losses.append(loss.detach().cpu().numpy())

                    loss.backward()

                    self.optim.step()

                    del sample_t, target_t, features_t, features_t_plus1, features_t_minus1

                del sample, target
                torch.cuda.empty_cache()

            print(f"\nEpoch: {epoch}|{config['training']['epochs']}\t\t Avg. loss: {np.mean(losses)}")
            self.writer.add_scalar(tag="train/loss", scalar_value=np.mean(losses), global_step=epoch)

            if not epoch % config['validation']['freq']:
                self.smoothness_penalty = False
                self.validate(training_epoch=epoch, config=config)
                self.smoothness_penalty = True

    def evaluate(self, dataset: Dataset = None, config: dict = None):

        batch_size = config['evaluation']['batch_size']
        assert batch_size == 1

        if dataset is not None:
            self.eval_data_loader = DataLoader(dataset=dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

        self.load_checkpoint(config['evaluation']['chckpt_path'])

        self.model.eval()

        with torch.no_grad():
            for i, sample in enumerate(self.eval_data_loader):

                sample, target = self.preprocess(sample, config)
                sample, target = sample.to(self.device), target.to(self.device)

                # Use time-steps as batch (Input tensor in format (T, C, H, W)
                prediction = self.model(sample.squeeze())

                play_video(prediction)

                print(f"Prediction {i}\t"
                      f"Shape {prediction.shape}\t"
                      f"Mean {prediction.mean()}\t"
                      f"Min {prediction.min()}\t"
                      f"Max {prediction.max()}")


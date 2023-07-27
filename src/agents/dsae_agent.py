""" Agent implementation for the Deep Spatial Auto-Encoder as used in
    https://arxiv.org/pdf/1509.06113.pdf


    # TODO: Make sure, batch sizes are handled correctly, i.e. tensors in (N, T, C, H, W) format
"""
import random
import gc

import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import mse_loss, interpolate
from torchvision.transforms.functional import rgb_to_grayscale

from src.utils.grad_flow import plot_grad_flow
from src.utils.kpt_utils import kpts_2_img_coordinates
from src.models.utils import init_weights
from .abstract_agent import AbstractAgent
from ..models.deep_spatial_autoencoder import DeepSpatialAE
from ..data.utils import play_video


class DSAEAgent(AbstractAgent):

    def __init__(self,
                 dataset: Dataset = None,
                 config: dict = None):

        super(DSAEAgent, self).__init__("Deep Spatial Auto-Encoder",
                                        dataset=dataset,
                                        config=config)

        self.smoothness_penalty = config['training']['smoothness_penalty']

        self.model = DeepSpatialAE(config['model'], device=config['device']).to(self.device)
        self.model.conv2.apply(lambda model: init_weights(m=model, config=config))
        self.model.conv3.apply(lambda model: init_weights(m=model, config=config))
        self.model.decoder.apply(lambda model: init_weights(m=model, config=config))

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

        loss = mse_loss(input=prediction, target=target, reduction='sum')

        if self.smoothness_penalty and self.model.training:
            penalty = mse_loss(ft_plus1 - ft, ft - ft_minus1, reduction='sum')
        else:
            penalty = torch.zeros(1, device=prediction.device)

        return loss + penalty

    def preprocess(self,
                   x: torch.Tensor,
                   label: torch.Tensor,
                   config: dict) -> (torch.Tensor, torch.Tensor):
        """ Returns sample and target from sampled data.

            The sampled image series is down-sampled and
            transformed to greyscale to give the target.
        """
        # img_series = x

        # Standardisation
        img_series = (x - x.mean())/x.std()

        target = interpolate(img_series,
                             size=(3,
                                   config['model']['fc']['out_img_width'],
                                   config['model']['fc']['out_img_height']
                                   )
                             )
        target = rgb_to_grayscale(target)

        return img_series, target

    def step(self,
             sample: torch.Tensor,
             target: torch.Tensor,
             global_epoch_number: int,
             save_grad_flow_plot: bool,
             save_val_sample: bool,
             config: dict,
             mode: str) -> torch.Tensor:

        timesteps = sample.shape[1]

        t = random.randint(1, sample.shape[1] - 2)

        if mode == 'training':
            self.optim.zero_grad()

        sample_t = sample[:, t, ...].to(self.device).unsqueeze(1)
        target_t = target[:, t, ...].to(self.device).unsqueeze(1)

        # Forward pass
        prediction = self.model(sample_t)

        assert prediction.shape == target_t.shape, \
            f"Prediction shape {prediction.shape} does not match " \
            f"target shape {target[t].unsqueeze(0).shape}"

        # Loss
        #with torch.no_grad():
        features_t_minus1, _ = self.model.encode(sample[:, t-1, ...].unsqueeze(1)) if t > 0 else \
            self.model.encode(sample_t)

        features_t, fmaps_t = self.model.encode(sample_t)  # (N, T, K, 2)

        features_t_plus1, _ = self.model.encode(sample[:, t+1, ...].unsqueeze(1)) if t < timesteps-1 else \
            self.model.encode(sample_t)

        loss = self.loss_func(prediction=prediction,
                              target=target_t,
                              ft_minus1=features_t_minus1,
                              ft=features_t,
                              ft_plus1=features_t_plus1)

        if mode == 'training':
            # Backward pass
            loss.backward()
            self.optim.step()

            if save_grad_flow_plot and config['training']['save_grad_flow_plot']:
                plot_grad_flow(named_parameters=self.model.named_parameters(),
                               epoch=global_epoch_number,
                               tag_name='train/grads',
                               summary_writer=self.writer)

            del sample_t, target_t, prediction, features_t_minus1, features_t, features_t_plus1

        if mode == 'validation' and config['validation']['save_plot'] and save_val_sample:

            fig, ax = plt.subplots(1, 2, figsize=(15, 5))
            ax[0].imshow(prediction[0].cpu().squeeze(), cmap='gray')
            ax[1].imshow(target_t[0].cpu().squeeze(), cmap='gray')

            self.writer.add_figure(tag='val/rec',
                                   figure=fig,
                                   global_step=global_epoch_number)

            plt.close()
            del fig, ax, prediction, target_t

            _features_t = torch.clone(features_t)
            _features_t[..., 0] *= -1

            img_coordinates_t = kpts_2_img_coordinates(_features_t[0, 0, ...], img_shape=sample_t[0].shape[1:])
            fig, ax = plt.subplots(1, 1, figsize=(15, 5))
            ax.imshow(sample_t[0].cpu().squeeze().permute(1, 2, 0), cmap='gray')
            ax.scatter(img_coordinates_t[:, 0].cpu(), img_coordinates_t[:, 1].cpu(), color='lime')

            self.writer.add_figure(tag='val/sample + kpts',
                                   figure=fig,
                                   global_step=global_epoch_number)

            plt.close()
            del fig, ax, sample_t, features_t, img_coordinates_t

            _fmaps_t = torch.clone(fmaps_t)  # (N, C, H', W')
            img_coordinates_t = kpts_2_img_coordinates(_features_t, _fmaps_t.shape[2:])  # (N, T, C, 2)
            fig, ax = plt.subplots(1, _fmaps_t[0].shape[0], figsize=(_fmaps_t[0].shape[0] * 3, 3))
            for c in range(_fmaps_t[0].shape[0]):
                ax[c].imshow(_fmaps_t[0, c, ...].cpu(), cmap='gray')
                ax[c].scatter(img_coordinates_t[0, 0, c, 0].cpu(),
                              img_coordinates_t[0, 0, c, 1].cpu(),
                              color='lime', marker='x')

            plt.tight_layout()
            self.writer.add_figure(tag='val/fmaps + kpts',
                                   figure=fig,
                                   global_step=global_epoch_number)

            plt.close()
            del fmaps_t, _fmaps_t, _features_t

            self.writer.flush()

            """
            for obj in gc.get_objects():
                try:
                    if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                        print(type(obj), obj.size())
                except:
                    pass
            """

        return loss

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

                sample, target = self.preprocess(sample, torch.empty([]), config)
                sample, target = sample.to(self.device), target.to(self.device)

                # Use time-steps as batch (Input tensor in format (T, C, H, W)
                print(sample.shape)
                prediction = self.model(sample)

                play_video(prediction[0, ...])

                print(f"Prediction {i}\t"
                      f"Shape {prediction.shape}\t"
                      f"Mean {prediction.mean()}\t"
                      f"Min {prediction.min()}\t"
                      f"Max {prediction.max()}")


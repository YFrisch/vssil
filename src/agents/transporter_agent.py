import random

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.models.transporter import Transporter
from src.models.inception3 import perception_inception_net
from src.models.alexnet import perception_alex_net
from src.utils.visualization import gen_eval_imgs
from src.utils.grad_flow import plot_grad_flow
from src.losses import perception_loss
from .abstract_agent import AbstractAgent


class TransporterAgent(AbstractAgent):

    def __init__(self,
                 dataset: Dataset,
                 config: dict):

        super(TransporterAgent, self).__init__(
            name='Transporter Agent',
            dataset=dataset,
            config=config
        )

        if config['training']['loss_function'] in ['inception', 'Inception', 'INCEPTION']:
            self.perception_net = perception_inception_net(config['log_dir']).to(self.device)

        if config['training']['loss_function'] in ['alexnet', 'AlexNet', 'ALEXNET']:
            self.perception_net = perception_alex_net(config['log_dir']).to(self.device)

        self.model = Transporter(config).to(self.device)

    def log_model(self, config: dict):
        pass
        """
        self.writer.add_text(tag='model/encoder',
                             text_string=str(summary(self.model.encoder, eval(config['data']['img_shape']))))
        self.writer.add_text(tag='model/decoder',
                             text_string=str(summary(self.model.decoder, eval(config['data']['img_shape']))))
        self.writer.add_text(tag='model/keypointer',
                             text_string=str(summary(self.model.keypointer, eval(config['data']['img_shape']))))
        """

    def preprocess(self,
                   x: torch.Tensor,
                   label: torch.Tensor,
                   config: dict) -> (torch.Tensor, (torch.Tensor, torch.Tensor)):

        """ Uses the first frame of a sequence of images as source and the last frame as target.

        :param x: Torch image series tensor in (N, T, C, H, W)
        :param label: -
        :param config: Additional args
        :return: Source frame in (N, C, H, W), target frame in (N, C, H, W)
        """

        assert x.dim() == 5

        # Randomly sample target frame from within t=s+1 to t=s+19
        # t_diff = random.randint(a=1, b=19)
        t_diff = random.randint(a=1, b=config['model']['n_frames'] - 1)

        # sample_frame = x[:, 0, ...] - 0.5
        # sample_frame = x[:, 0, ...]
        # target_frame = x[:, -1, ...] - 0.5
        # target_frame = x[:, -1, ...]
        # target_frame = x[:, 0 + t_diff, ...]
        # Normalizing to [-1, 1] range
        sample_frame = 2 * ((x[:, 0, ...] - x[:, 0, ...].min()) / (x[:, 0, ...].max() - x[:, 0, ...].min())) - 1
        target_frame = 2 * ((x[:, 0 + t_diff, ...] - x[:, 0 + t_diff, ...].min()) /
                            (x[:, 0 + t_diff, ...].max() - x[:, 0 + t_diff, ...].min())) - 1
        return sample_frame, target_frame

    def loss_func(self, prediction: torch.Tensor, target: torch.Tensor, config: dict) -> torch.Tensor:
        if config['training']['loss_function'] in ['mse', 'l2', 'MSE']:
            return F.mse_loss(input=prediction, target=target, reduction='mean')
        elif config['training']['loss_function'] in ['sse', 'SSE']:
            # return F.mse_loss(input=prediction, target=target, reduction='sum')
            return F.mse_loss(prediction, target, reduction='sum') / \
                   (torch.var(target) * 2 * target.shape[0] * target.shape[1])
        elif config['training']['loss_function'] in ['bce', 'BCE']:
            # Map to [0, 1]
            prediction = (prediction + 1.0) / 2.0
            target = (target + 1.0) / 2.0
            return F.binary_cross_entropy(input=prediction, target=target)
        elif config['training']['loss_function'] in ['alexnet', 'AlexNet', 'ALEXNET',
                                                     'inception', 'Inception', 'INCEPTION']:
            # Map to [0, 1]
            prediction = (prediction + 1.0) / 2.0
            target = (target + 1.0) / 2.0
            loss_perception = perception_loss(perception_net=self.perception_net,
                                              prediction=prediction.unsqueeze(1),
                                              target=target.unsqueeze(1))
            # loss_pixel = F.binary_cross_entropy(input=prediction, target=target)
            loss_pixel = F.mse_loss(input=prediction, target=target, reduction='mean')
            return 0.5 * loss_perception + 0.5 * loss_pixel
        else:
            raise ValueError("Given loss function not implemented.")

    def step(self,
             sample: torch.Tensor,
             target: torch.Tensor,
             global_epoch_number: int,
             save_grad_flow_plot: bool,
             save_val_sample: bool,
             config: dict, mode: str) -> torch.Tensor:

        assert mode in ['training', 'validation']

        N, C, H, W = sample.shape

        if mode == 'training':
            self.optim.zero_grad()

        reconstruction = self.model(sample, target)
        # reconstruction.clip_(-0.5, 0.5)
        loss = self.loss_func(prediction=reconstruction, target=target, config=config)

        if mode == 'training':

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config['training']['grad_clip_max_norm'])

            if save_grad_flow_plot:
                plot_grad_flow(named_parameters=self.model.encoder.named_parameters(),
                               epoch=global_epoch_number,
                               tag_name='encoder',
                               summary_writer=self.writer)
                plot_grad_flow(named_parameters=self.model.keypointer.named_parameters(),
                               epoch=global_epoch_number,
                               tag_name='keypointer',
                               summary_writer=self.writer)
                plot_grad_flow(named_parameters=self.model.decoder.named_parameters(),
                               epoch=global_epoch_number,
                               tag_name='decoder',
                               summary_writer=self.writer)
            self.optim.step()

        if mode == 'validation' and config['validation']['save_eval_examples']:
            with torch.no_grad():
                key_point_coordinates = torch.cat([self.model.keypointer(sample)[0].unsqueeze(1),
                                                   self.model.keypointer(target)[0].unsqueeze(1)], dim=1)
                # Adapt to visualization
                key_point_coordinates[..., 1] = key_point_coordinates[..., 1] * (-1)
                _sample = torch.cat([sample.unsqueeze(1), target.unsqueeze(1)], dim=1)
                _sample = ((_sample + 1) / 2.0).clip(0.0, 1.0)
                reconstruction = torch.cat([sample.unsqueeze(1), reconstruction.unsqueeze(1)], dim=1)
                reconstruction = ((reconstruction + 1) / 2.0).clip(0.0, 1.0)
                torch_img_series_tensor = gen_eval_imgs(sample=_sample,
                                                        reconstruction=reconstruction,
                                                        key_points=key_point_coordinates)

                self.writer.add_video(tag='val/reconstruction_sample',
                                      vid_tensor=torch_img_series_tensor,
                                      global_step=global_epoch_number)
                self.writer.flush()

        return loss

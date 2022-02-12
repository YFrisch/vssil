import random

import pylab
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.utils import make_grid
from torchvision import transforms

from src.models.transporter import Transporter
from src.models.inception3 import perception_inception_net
from src.models.alexnet import perception_alex_net
from src.utils.visualization import gen_eval_imgs
from src.utils.grad_flow import plot_grad_flow
from src.utils.kpt_utils import kpts_2_img_coordinates
from src.losses import perception_loss
from .abstract_agent import AbstractAgent

Tran = transforms.RandomApply([
    transforms.RandomHorizontalFlip(p=0.3),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=45, )
])


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
        # TODO: The Human36M Data seems to work better in [0, 1] range instead...
        # sample_frame = 2 * ((x[:, 0, ...] - x[:, 0, ...].min()) / (x[:, 0, ...].max() - x[:, 0, ...].min())) - 1
        # target_frame = 2 * ((x[:, 0 + t_diff, ...] - x[:, 0 + t_diff, ...].min()) /
        #                   (x[:, 0 + t_diff, ...].max() - x[:, 0 + t_diff, ...].min())) - 1

        # sample_frame = ((x[:, 0, ...] - x[:, 0, ...].min()) / (x[:, 0, ...].max() - x[:, 0, ...].min()))
        # target_frame = ((x[:, 0 + t_diff, ...] - x[:, 0 + t_diff, ...].min()) /
        #                (x[:, 0 + t_diff, ...].max() - x[:, 0 + t_diff, ...].min()))

        sample_frame = x[:, 0, ...]
        target_frame = x[:, 0 + t_diff, ...]
        # target_frame = Tran(x[:, 0 + t_diff, ...])

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
            # prediction = (prediction + 1.0) / 2.0
            # target = (target + 1.0) / 2.0
            return F.binary_cross_entropy(input=prediction, target=target)
        elif config['training']['loss_function'] in ['alexnet', 'AlexNet', 'ALEXNET',
                                                     'inception', 'Inception', 'INCEPTION']:
            # Map to [0, 1]
            # prediction = (prediction + 1.0) / 2.0
            # target = (target + 1.0) / 2.0
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

        self.optim.zero_grad()

        reconstruction = self.model(sample, target)

        loss = self.loss_func(prediction=reconstruction, target=target, config=config)

        if mode == 'training':

            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config['training']['grad_clip_max_norm'])

            self.optim.step()

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

        if mode == 'validation' and config['validation']['save_eval_examples'] and save_val_sample:

            with torch.no_grad():
                source_kpts, source_gmaps, source_fmaps = self.model.keypointer(sample)
                _N, _K, _H, _W = source_fmaps.shape
                target_kpts, target_gmaps, target_fmaps = self.model.keypointer(target)
                key_point_coordinates = torch.cat([source_kpts.unsqueeze(1), target_kpts.unsqueeze(1)], dim=1)

                # Adapt to visualization
                key_point_coordinates[..., 0] *= -1.0

                # Convert kpt coordinates from [-1, 1]x[-1, 1] to [0, H]x[0, W]
                img_coordinates = kpts_2_img_coordinates(key_point_coordinates, (_H, _W)).cpu()  # (N, 2, K, 2)
                key_point_coordinates[..., :2] *= -1.0

                _sample = torch.cat([sample.unsqueeze(1), target.unsqueeze(1)], dim=1)
                # _sample = ((_sample + 1) / 2.0).clip(0.0, 1.0)

                reconstruction = torch.cat([sample.unsqueeze(1), reconstruction.unsqueeze(1)], dim=1)
                # reconstruction = ((reconstruction + 1) / 2.0).clip(0.0, 1.0)

                torch_img_series_tensor = gen_eval_imgs(sample=_sample,
                                                        reconstruction=reconstruction,
                                                        key_points=key_point_coordinates)

                self.writer.add_video(tag='val/reconstruction_sample',
                                      vid_tensor=torch_img_series_tensor,
                                      global_step=global_epoch_number)

                cm = pylab.get_cmap('gist_rainbow')

                fig, ax = plt.subplots(2, _K, figsize=(_K * 4, 8))
                for _k in range(_K):
                    ax[0, _k].imshow(source_fmaps[0, _k, ...].cpu(), cmap='gray')
                    ax[0, _k].set_title(f'Source frame - Keypoint {_k}')
                    ax[0, _k].scatter(img_coordinates[0, 0, _k, 0], img_coordinates[0, 0, _k, 1],
                                      color=cm(1. * _k / _K), marker="^", s=150)

                    max_loc_source = torch.argmax(source_fmaps[0, _k, ...]).cpu()
                    max_loc_source_h = max_loc_source / source_fmaps[0, _k, ...].shape[-2]
                    max_loc_source_w = max_loc_source % source_fmaps[0, _k, ...].shape[-1]
                    ax[0, _k].scatter(max_loc_source_w, max_loc_source_h,
                                      color='red', marker="x", s=250)

                    ax[1, _k].imshow(target_fmaps[0, _k, ...].cpu(), cmap='gray')
                    ax[1, _k].set_title(f'Target frame - Keypoint {_k}')
                    ax[1, _k].scatter(img_coordinates[0, 1, _k, 0], img_coordinates[0, 1, _k, 1],
                                      color=cm(1. * _k / _K), marker="^", s=150)

                    max_loc_target = torch.argmax(target_fmaps[0, _k, ...]).cpu()
                    max_loc_target_h = max_loc_target / target_fmaps[0, _k, ...].shape[-2]
                    max_loc_target_w = max_loc_target % target_fmaps[0, _k, ...].shape[-1]
                    ax[1, _k].scatter(max_loc_target_w, max_loc_target_h,
                                      color='red', marker="x", s=250)

                plt.tight_layout()
                self.writer.add_figure(tag="val/feature_maps",
                                       figure=fig,
                                       global_step=global_epoch_number)

                fig, ax = plt.subplots(2, _K, figsize=(_K * 4, 8))
                for _k in range(_K):
                    ax[0, _k].imshow(source_gmaps[0, _k, ...].cpu(), cmap='gray')
                    ax[0, _k].set_title(f'Source frame - Keypoint {_k}')
                    ax[0, _k].scatter(img_coordinates[0, 0, _k, 0], img_coordinates[0, 0, _k, 1],
                                      color=cm(1. * _k / _K), marker="^", s=150)

                    ax[1, _k].imshow(target_gmaps[0, _k, ...].cpu(), cmap='gray')
                    ax[1, _k].set_title(f'Target frame - Keypoint {_k}')
                    ax[1, _k].scatter(img_coordinates[0, 1, _k, 0], img_coordinates[0, 1, _k, 1],
                                      color=cm(1. * _k / _K), marker="^", s=150)



                plt.tight_layout()
                self.writer.add_figure(tag="val/gaussian_maps",
                                       figure=fig,
                                       global_step=global_epoch_number)

                plt.close()

                self.writer.flush()

        return loss

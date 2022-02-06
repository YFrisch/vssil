import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
import pylab
import matplotlib.pyplot as plt

from src.models.ulosd import ULOSD, ULOSD_Parallel, ULOSD_Dist_Parallel
from src.models.inception3 import perception_inception_net
from src.models.alexnet import perception_alex_net
from src.losses import temporal_separation_loss, perception_loss, spatial_consistency_loss, \
    time_contrastive_triplet_loss, pixelwise_contrastive_loss_patch_based, pixelwise_contrastive_loss_fmap_based, \
    pwcl, pwcl2
from src.losses.distr_constraints import wasserstein_constraint
from src.utils.grad_flow import plot_grad_flow
from src.utils.visualization import gen_eval_imgs
from src.utils.kpt_utils import kpts_2_img_coordinates
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

        # Input shape
        N = config['training']['batch_size']
        T = config['model']['n_frames']
        C = 3
        H = eval(config['data']['img_shape'])[0]
        W = eval(config['data']['img_shape'])[1]
        input_shape = (N, T, C, H, W)

        self.model = ULOSD(input_shape=input_shape, config=config).to(self.device)

        if config['training']['reconstruction_loss'] in ['inception', 'Inception', 'INCEPTION']:
            self.perception_net = perception_inception_net(config['log_dir']).to(self.device)

        if config['training']['reconstruction_loss'] in ['alexnet', 'AlexNet', 'ALEXNET']:
            self.perception_net = perception_alex_net(config['log_dir']).to(self.device)

        if config['multi_gpu'] is True and torch.cuda.device_count() >= 1:
            # self.model = ULOSD_Dist_Parallel(
            self.model = ULOSD_Parallel(
                module=self.model,
                device_ids=list(range(torch.cuda.device_count())),
                dim=0
            )
            # self.model = ULOSD_Parallel(self.model)
            # TODO: Is the next line required?
            self.model.to(self.device)

        # Properties for patch-wise contrastive loss
        if config['training']['patchwise_contrastive_scale'] != 0.0:
            K = config['model']['n_feature_maps']
            pc_time_window = config['training']['patchwise_contrastive_time_window']
            pc_patch_size = eval(config['training']['patchwise_contrastive_patch_size'])
            assert pc_time_window <= T
            assert pc_time_window % 2 != 0, "Use odd time-window"
            assert pc_patch_size[0] == pc_patch_size[1], "Use square patch"
            assert config['training']['batch_size'] == config['validation']['batch_size']
            self.pc_pos_range = max(int(pc_time_window / 2), 1) if pc_time_window > 1 else 0
            pc_center_index = int(pc_patch_size[0] / 2)
            pc_step_matrix = torch.ones(pc_patch_size + (2,)).to(self.device)
            # TODO: Check step sizes
            step_w = 1 / W
            step_h = 1 / H
            for k in range(0, pc_patch_size[0]):
                for l in range(0, pc_patch_size[1]):
                    pc_step_matrix[k, l, 0] = (l - pc_center_index) * step_w
                    pc_step_matrix[k, l, 1] = (k - pc_center_index) * step_h

            self.pc_grid = pc_step_matrix.unsqueeze(0).repeat((N * T * K, 1, 1, 1)).to(self.device)

        # Logged values
        self.rec_loss_per_iter = []
        self.sep_loss_per_iter = []
        self.pacolo_per_iter = []  # Extension
        self.match_loss_per_iter = []  # Extension
        self.non_match_loss_per_iter = []  # Extension
        self.emd_sum_per_iter = []  # Extension
        self.l1_penalty_per_iter = []
        self.total_loss_per_iter = []

        self.val_rec_loss = []
        self.val_sep_loss = []
        self.val_sparsity_loss = []
        self.val_pacolo = []
        self.val_match_loss = []
        self.val_non_match_loss = []

    def preprocess(self,
                   x: torch.Tensor,
                   label: torch.Tensor,
                   config: dict) -> (torch.Tensor, (torch.Tensor, torch.Tensor)):
        """ Maps the input image sequence to range (-0.5, 0.5) [0.0, 1.0].
            Then returns input as input and target for reconstruction.

            NOTE: MIME data should already be in (0, 1).

        :param x: Image sequence in (N, T, C, H, W)
        :param label: Label sequence in (N, T, K)
        :param config:
        :return:
        """
        # assert x.max() <= 1
        # assert x.min() >= 0
        # TODO: Check if range [0, 1] works better than [-0.5, 0.5]
        # x = torch.clamp(x - 0.5, min=-0.5, max=0.5)
        # x = torch.clamp(x, min=0.0, max=1.0)
        # x = ((x - x.min()) / (x.max() - x.min()) - 0.5).clamp(-0.5, 0.5)
        # x = (2 * ((x - x.min()) / (x.max() - x.min())) - 1.0).clamp(-1.0, 1.0)
        return x, torch.empty([])

    def loss_func(self,
                  prediction: torch.Tensor,
                  target: torch.Tensor,
                  config: dict) -> torch.Tensor:
        """ Loss for image reconstruction.

            Either uses the (normalized) L2 MSE as in
            ...TODO
            or passes the original image and the reconstruction trough a port
            of a pretrained inception net and calculates the loss on that encoding
            as in
            ...TODO

            NOTE: PyTorch already averages across the first dimension by default (N).

        :param prediction: Sequence of predicted images in (N, T, C, H, W)
        :param target: Actual image sequence in (N, T, C, H, W)
        :param config: Config dictionary
        :return: Reconstruction loss between prediction and target
        """
        rec_loss = config['training']['reconstruction_loss']

        N, T = target.shape[0], target.shape[1]

        loss = None

        if rec_loss in ['mse', 'MSE']:
            loss = F.mse_loss(input=prediction, target=target, reduction='mean') * 0.5
            # loss /= (N*T)

        elif rec_loss in ['sse', 'SSE']:
            loss = F.mse_loss(input=prediction, target=target, reduction='sum') * 0.5
            loss = loss / (N * T)

        elif rec_loss in ['bce', 'BCE']:
            loss = F.binary_cross_entropy(input=prediction, target=target, reduction='mean')

        elif rec_loss in ['Inception', 'inception', 'INCEPTION', 'alexnet', 'AlexNet', 'ALEXNET']:
            loss = perception_loss(perception_net=self.perception_net,
                                   prediction=prediction,
                                   target=target)
            # loss = loss / (N * T)

        else:
            raise ValueError("Unknown error function.")

        return loss * config['training']['reconstruction_loss_scale']

    def separation_loss(self, keypoint_coordinates: torch.Tensor, config: dict) -> torch.Tensor:
        separation_loss_scale = config['training']['separation_loss_scale']
        return temporal_separation_loss(cfg=config, coords=keypoint_coordinates) * separation_loss_scale

    def patchwise_contrastive_loss(self,
                                   keypoint_coordinates: torch.Tensor,
                                   feature_map_sequence: torch.Tensor,
                                   image_sequence: torch.Tensor,
                                   config: dict) -> torch.Tensor:
        scale = config['training']['patchwise_contrastive_scale']
        if config['training']['patchwise_contrastive_type'] == 'patch':
            pacolo, match_l, non_match_l = pwcl2(
                keypoint_coordinates=keypoint_coordinates,
                image_sequence=image_sequence,
                grid=self.pc_grid,
                patch_size=eval(config['training']['patchwise_contrastive_patch_size']),
                alpha=config['training']['patchwise_contrastive_alpha'],
            )
            return pacolo * scale, match_l, non_match_l
        elif config['training']['patchwise_contrastive_type'] == 'fmap':
            return pixelwise_contrastive_loss_fmap_based(
                keypoint_coordinates=keypoint_coordinates,
                image_sequence=image_sequence,
                feature_map_sequence=feature_map_sequence,
                pos_range=self.pc_pos_range,
                alpha=config['training']['patchwise_contrastive_alpha'],
            ) * scale
        else:
            raise ValueError(f"Unknown pc loss type: {config['training']['patchwise_contrastive_type']}")

    def l1_activation_penalty(self, feature_maps: torch.Tensor, config: dict) -> torch.Tensor:
        feature_map_mean = torch.mean(feature_maps, dim=[-2, -1])
        penalty = torch.mean(torch.abs(feature_map_mean))
        return config['training']['feature_map_regularization'] * penalty

    def key_point_sparsity_loss(self, keypoint_coordinates: torch.Tensor, config: dict) -> torch.Tensor:
        key_point_scales = keypoint_coordinates[..., 2]
        loss = torch.mean(torch.sum(torch.abs(key_point_scales), dim=2), dim=[0, 1])
        return config['training']['feature_map_regularization'] * loss

    def l2_kernel_regularization(self, config: dict) -> torch.Tensor:
        """ TODO: This is replaced by PyTorch's weight decay in the optimizer. """
        l2_reg = None

        for p_name, param in self.model.named_parameters():
            if ".weight" in p_name:
                if l2_reg is None:
                    l2_reg = param.norm(2) ** 2
                else:
                    l2_reg = l2_reg + param.norm(2) ** 2

        return config['training']['l2_kernel_reg_lambda'] * l2_reg

    def reset_logged_values(self):
        super(ULOSD_Agent, self).reset_logged_values()
        self.rec_loss_per_iter = []
        self.sep_loss_per_iter = []
        self.pacolo_per_iter = []  # Extension
        self.match_loss_per_iter = []  # Extension
        self.non_match_loss_per_iter = []  # Extension
        self.emd_sum_per_iter = []  # Extension
        self.l1_penalty_per_iter = []
        self.total_loss_per_iter = []

        self.val_rec_loss = []
        self.val_sep_loss = []
        self.val_sparsity_loss = []
        self.val_pacolo = []
        self.val_match_loss = []
        self.val_non_match_loss = []

    def log_values(self, fold: int, epoch: int, epochs_per_fold: int, mode: str = 'training'):

        global_epoch = fold * epochs_per_fold + epoch

        if mode == 'training':
            avg_reconstruction_loss = np.mean(self.rec_loss_per_iter)
            avg_separation_loss = np.mean(self.sep_loss_per_iter)
            avg_pacolo = np.mean(self.pacolo_per_iter)  # Extension
            avg_match_loss = np.mean(self.match_loss_per_iter)  # Extension
            avg_non_match_loss = np.mean(self.non_match_loss_per_iter)  # Extension
            avg_emd_sum = np.mean(self.emd_sum_per_iter)  # Extension
            avg_l1_penalty = np.mean(self.l1_penalty_per_iter)
            avg_total_loss = np.mean(self.total_loss_per_iter)

            self.writer.add_scalar(tag="train/reconstruction_loss",
                                   scalar_value=avg_reconstruction_loss, global_step=global_epoch)
            self.writer.add_scalar(tag="train/separation_loss",
                                   scalar_value=avg_separation_loss, global_step=global_epoch)
            self.writer.add_scalar(tag="train/patchwise_contrastive_loss",
                                   scalar_value=avg_pacolo, global_step=global_epoch)  # Extension
            self.writer.add_scalar(tag="train/pacolo_match_loss",
                                   scalar_value=avg_match_loss, global_step=global_epoch)  # Extension
            self.writer.add_scalar(tag="train/pacolo_non_match_loss",
                                   scalar_value=avg_non_match_loss, global_step=global_epoch)  # Extension
            # self.writer.add_scalar(tag="train/emd_sum",
            #                       scalar_value=avg_emd_sum, global_step=global_epoch)  # Extension
            self.writer.add_scalar(tag="train/l1_activation_penalty",
                                   scalar_value=avg_l1_penalty, global_step=global_epoch)
            self.writer.add_scalar(tag="train/total_loss",
                                   scalar_value=avg_total_loss, global_step=global_epoch)

        elif mode == 'validation':
            avg_val_rec_loss = np.mean(self.val_rec_loss)
            avg_val_sep_loss = np.mean(self.val_sep_loss)
            avg_val_sparsity_loss = np.mean(self.val_sparsity_loss)
            avg_val_pacolo = np.mean(self.val_pacolo)
            avg_val_match_loss = np.mean(self.val_match_loss)
            avg_val_non_match_loss = np.mean(self.val_non_match_loss)

            self.writer.add_scalar(tag="val/reconstruction_loss",
                                   scalar_value=avg_val_rec_loss, global_step=global_epoch)
            self.writer.add_scalar(tag="val/separation_loss",
                                   scalar_value=avg_val_sep_loss, global_step=global_epoch)
            self.writer.add_scalar(tag="val/l1_activation_penalty",
                                   scalar_value=avg_val_sparsity_loss, global_step=global_epoch)
            self.writer.add_scalar(tag="val/patchwise_contrastive_loss",
                                   scalar_value=avg_val_pacolo, global_step=global_epoch)
            self.writer.add_scalar(tag="val/pacolo_match_loss",
                                   scalar_value=avg_val_match_loss, global_step=global_epoch)
            self.writer.add_scalar(tag="val/pacolo_non_match_loss",
                                   scalar_value=avg_val_non_match_loss, global_step=global_epoch)

        else:
            raise ValueError('Unkown mode.')

    def step(self,
             sample: torch.Tensor,
             target: torch.Tensor,
             global_epoch_number: int,
             save_grad_flow_plot: bool,
             save_val_sample: bool,
             config: dict,
             mode: str) -> torch.Tensor:
        """ One step of training.

        :param sample: Image sequence in (N, T, C, H, W)
        :param target: -
        :param global_epoch_number: Number of global epochs
        :param save_grad_flow_plot: Whether or not to plot the gradient flow
        :param save_val_sample: Set true to save a sample video during validation
        :param config: Configuration dictionary
        :param mode: Flag to determine 'training' or 'validation' mode
        :return: Total loss
        """

        assert mode in ['training', 'validation']

        if mode == 'training':
            # self.optim.zero_grad(set_to_none=True)
            self.optim.zero_grad()

        #
        # Vision model
        #

        feature_maps, observed_key_points = self.model.encode(
            image_sequence=sample,
            appearance=False,
            re_sample_kpts=config['training']['re_sample'],
            re_sample_scale=config['training']['re_sample_scale'])
        N, T, C, Hp, Wp = feature_maps.shape
        _, _, K, D = observed_key_points.shape
        assert observed_key_points[..., :2].max() <= 1.0, f'{observed_key_points[..., :2].max()} > 1.0'

        # predicted_diff = self.model.decode(observed_key_points, sample[:, 0:1, ...])
        # reconstruction = sample[:, 0:1, ...] + predicted_diff
        reconstruction, gaussian_maps = self.model.decode(observed_key_points, sample[:, 0:1, ...])

        #
        # Dynamics model
        #

        # TODO: Not used yet

        #
        # Reconstruction loss for reconstructing the input image sequence
        #

        reconstruction_loss = self.loss_func(prediction=reconstruction, target=sample, config=config)

        #
        # Separation loss for key-point trajectories
        #

        separation_loss = self.separation_loss(keypoint_coordinates=observed_key_points, config=config)

        #
        # Feature-map (L1) regularization of the activations of the last layer
        # (Sparsity Loss)
        #

        l1_penalty = self.l1_activation_penalty(feature_maps=feature_maps, config=config)
        # l1_penalty = self.key_point_sparsity_loss(keypoint_coordinates=observed_key_points, config=config)

        #
        # Extensions
        #

        if config['training']['patchwise_contrastive_scale'] > 0 \
                and global_epoch_number >= config['training']['patchwise_contrastive_starting_epoch']:
            pc_loss, match_loss, non_match_loss = self.patchwise_contrastive_loss(
                keypoint_coordinates=observed_key_points,
                image_sequence=sample,
                feature_map_sequence=gaussian_maps,
                config=config)
        else:
            pc_loss = torch.Tensor([0.0]).to(self.device)
            match_loss = torch.Tensor([0.0]).to(self.device)
            non_match_loss = torch.Tensor([0.0]).to(self.device)

        if config['training']['use_emd']:
            emd_sum = (torch.sum(wasserstein_constraint(observed_key_points))
                       * config['training']['emd_sum_scale']).to(self.device)
        else:
            emd_sum = torch.Tensor([0.0]).to(self.device)

        #
        # Losses for the dynamics model
        # TODO: not used yet
        #

        coord_pred_loss = 0
        kl_loss = 0
        kl_loss_scale = 0
        kl_loss = kl_loss * kl_loss_scale

        # total loss
        L = reconstruction_loss + separation_loss + l1_penalty + \
            coord_pred_loss + kl_loss + \
            pc_loss + emd_sum

        if mode == 'validation':
            # NOTE: This part seems to cause a linear increase in CPU memory usage
            #       Maybe the videos should be saved to the hard-drive instead

            with torch.no_grad():

                self.val_rec_loss.append(reconstruction_loss.item())
                self.val_sep_loss.append(separation_loss.item())
                self.val_sparsity_loss.append(l1_penalty.item())
                self.val_pacolo.append(pc_loss.item())
                self.val_match_loss.append(match_loss.item())
                self.val_non_match_loss.append(non_match_loss.item())

                if config['validation']['save_video'] and save_val_sample:

                    torch_img_series_tensor = gen_eval_imgs(sample=sample,
                                                            reconstruction=reconstruction,
                                                            key_points=observed_key_points,
                                                            intensity_threshold=0.5)

                    self.writer.add_video(tag='val/reconstruction_sample',
                                          vid_tensor=torch_img_series_tensor,
                                          global_step=global_epoch_number)

                    cm = pylab.get_cmap('gist_rainbow')

                    observed_key_points[..., :2] *= -1.0
                    img_coordinates = kpts_2_img_coordinates(observed_key_points, (Hp, Wp)).cpu()
                    observed_key_points[..., :2] *= -1.0

                    t = T - 1
                    # Get ids of 'active' kpts (intensity above threshold)
                    active_kpt_ids = np.array([k if observed_key_points[0, t, k, 2] > 0.5 else 0 for k in range(K)])
                    active_kpt_ids = np.delete(active_kpt_ids, np.where(active_kpt_ids == 0))
                    _K = len(active_kpt_ids)

                    fig, ax = plt.subplots(2, _K, figsize=(_K * 3, 6))

                    for k in range(_K):

                        if _K == 1:

                            ax[0].imshow(feature_maps[0, t, active_kpt_ids[k], ...].cpu(), cmap='gray')
                            ax[0].set_title(f'Frame {t} - F. Map {active_kpt_ids[k]}')
                            ax[0].scatter(img_coordinates[0, t, active_kpt_ids[k], 1],
                                          img_coordinates[0, t, active_kpt_ids[k], 0],
                                          color=cm(1. * active_kpt_ids[k] / K),
                                          marker="^", s=150)

                            ax[1].imshow(gaussian_maps[0, t, active_kpt_ids[k], ...].cpu(), cmap='gray')
                            ax[1].set_title(f'Frame {t} - G. Rec. {active_kpt_ids[k]}')
                            ax[1].scatter(img_coordinates[0, t, active_kpt_ids[k], 1],
                                          img_coordinates[0, t, active_kpt_ids[k], 0],
                                          color=cm(1. * active_kpt_ids[k] / K), marker="^", s=150)

                        else:

                            ax[0, k].imshow(feature_maps[0, t, active_kpt_ids[k], ...].cpu(), cmap='gray')
                            ax[0, k].set_title(f'Frame {t} - F. Map {active_kpt_ids[k]}')
                            ax[0, k].scatter(img_coordinates[0, t, active_kpt_ids[k], 1],
                                             img_coordinates[0, t, active_kpt_ids[k], 0],
                                             color=cm(1. * active_kpt_ids[k] / K),
                                             marker="^", s=150)

                            ax[1, k].imshow(gaussian_maps[0, t, active_kpt_ids[k], ...].cpu(), cmap='gray')
                            ax[1, k].set_title(f'Frame {t} - G. Rec. {active_kpt_ids[k]}')
                            ax[1, k].scatter(img_coordinates[0, t, active_kpt_ids[k], 1],
                                             img_coordinates[0, t, active_kpt_ids[k], 0],
                                             color=cm(1. * active_kpt_ids[k] / K), marker="^", s=150)

                    self.writer.add_figure(tag="feature_maps_and_reconstructions",
                                           figure=fig,
                                           global_step=global_epoch_number)

                    plt.close()

                    self.writer.flush()
                    del torch_img_series_tensor

        # Log values and backprop. during training
        if mode == 'training':

            self.rec_loss_per_iter.append(reconstruction_loss.item())
            self.sep_loss_per_iter.append(separation_loss.item())
            self.pacolo_per_iter.append(pc_loss.item())  # Extension
            self.match_loss_per_iter.append(match_loss.item())  # Extension
            self.non_match_loss_per_iter.append(non_match_loss.item())  # Extension
            self.emd_sum_per_iter.append(emd_sum.item())  # Extension
            self.l1_penalty_per_iter.append(l1_penalty.item())
            self.total_loss_per_iter.append(L.item())

            L.backward()

            # Clip gradient norm
            nn.utils.clip_grad_norm_(self.model.parameters(), config['training']['clip_norm'])

            with torch.no_grad():
                if save_grad_flow_plot:
                    plot_grad_flow(named_parameters=self.model.encoder.named_parameters(),
                                   epoch=global_epoch_number,
                                   summary_writer=self.writer,
                                   tag_name='encoder')
                    plot_grad_flow(named_parameters=self.model.appearance_net.named_parameters(),
                                   epoch=global_epoch_number,
                                   summary_writer=self.writer,
                                   tag_name='appearance_net')
                    plot_grad_flow(named_parameters=self.model.decoder.named_parameters(),
                                   epoch=global_epoch_number,
                                   summary_writer=self.writer,
                                   tag_name='decoder')
                    self.writer.flush()

            self.optim.step()

        return L

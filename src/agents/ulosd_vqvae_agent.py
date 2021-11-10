import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.losses import pc_loss, temporal_separation_loss
from src.utils.grad_flow import plot_grad_flow
from src.utils.visualization import gen_eval_imgs
from src.models.vqvae import VQ_VAE_KPT
from .abstract_agent import AbstractAgent


class VQVAE_Agent(AbstractAgent):

    def __init__(self,
                 dataset: Dataset,
                 config: dict):
        super(VQVAE_Agent, self).__init__(name='VQ-VAE Agent', dataset=dataset, config=config)

        # Input shape
        N = config['training']['batch_size']
        T = config['model']['n_frames']
        C = 3
        H = eval(config['data']['img_shape'])[0]
        W = eval(config['data']['img_shape'])[1]
        input_shape = (T, C, H, W)

        self.model = VQ_VAE_KPT(
            batch_size=N,
            time_steps=T,
            num_embeddings=config['model']['n_codebook_embeddings'],
            heatmap_width=config['model']['heatmap_width'],
            encoder_in_channels=C,
            num_hiddens=config['model']['num_hiddens'],
            embedding_dim=config['model']['embedding_dim'],
            num_residual_layers=config['model']['n_residual_layers'],
            num_residual_hiddens=config['model']['n_residual_hiddens'],
            device=config['device']
        ).to(config['device'])

        # Logged values
        self.total_loss_per_iter = []
        self.rec_loss_per_iter = []
        self.sep_loss_per_iter = []
        self.sparsity_loss_per_iter = []
        self.rec_vq_loss_per_iter = []
        self.rec_g_loss_per_iter = []
        self.vq_loss_per_iter = []
        self.pc_loss_per_iter = []
        self.perplexity_per_iter = []

    def reset_logged_values(self):
        super(VQVAE_Agent, self).reset_logged_values()
        self.total_loss_per_iter = []
        self.rec_loss_per_iter = []
        self.sep_loss_per_iter = []
        self.sparsity_loss_per_iter = []
        self.rec_vq_loss_per_iter = []
        self.rec_g_loss_per_iter = []
        self.vq_loss_per_iter = []
        self.pc_loss_per_iter = []
        self.perplexity_per_iter = []

    def log_values(self,
                   fold: int,
                   epoch: int,
                   epochs_per_fold: int):
        global_epoch = fold * epochs_per_fold + epoch
        avg_total_loss = np.mean(self.total_loss_per_iter)
        avg_rec_loss = np.mean(self.rec_loss_per_iter)
        avg_sep_loss = np.mean(self.sep_loss_per_iter)
        avg_sparsity_loss = np.mean(self.sparsity_loss_per_iter)
        avg_vq_loss = np.mean(self.vq_loss_per_iter)
        avg_pc_loss = np.mean(self.pc_loss_per_iter)
        avg_perplexity = np.mean(self.perplexity_per_iter)
        self.writer.add_scalar(tag="train/total_loss",
                               scalar_value=avg_total_loss, global_step=global_epoch)
        self.writer.add_scalar(tag="train/reconstruction_loss",
                               scalar_value=avg_rec_loss, global_step=global_epoch)
        self.writer.add_scalar(tag="train/temporal_separation_loss",
                               scalar_value=avg_sep_loss, global_step=global_epoch)
        self.writer.add_scalar(tag="train/keypoint_sparsity_loss",
                               scalar_value=avg_sparsity_loss, global_step=global_epoch)
        self.writer.add_scalar(tag="train/vq_loss",
                               scalar_value=avg_vq_loss, global_step=global_epoch)
        self.writer.add_scalar(tag="train/perplexity",
                               scalar_value=avg_perplexity, global_step=global_epoch)
        self.writer.add_scalar(tag="train/pixelwise_contrastive_loss",
                               scalar_value=avg_pc_loss, global_step=global_epoch)

    def preprocess(self,
                   x: torch.Tensor,
                   label: torch.Tensor,
                   config: dict) -> (torch.Tensor, (torch.Tensor, torch.Tensor)):
        """ Maps the input image sequence to range (-0.5, 0.5).
            Then returns input as input and target for reconstruction.

            NOTE: MIME data should already be in (0, 1).

        :param x: Image sequence in (N, T, C, H, W)
        :param label: Label sequence in (N, T, K)
        :param config:
        :return:
        """
        assert x.max() <= 1
        assert x.min() >= 0
        x = torch.clamp(x - 0.5, min=-0.5, max=0.5)
        return x, torch.empty([])

    def loss_func(self, prediction: torch.Tensor, target: torch.Tensor, config: dict) -> torch.Tensor:
        """ Reconstruction loss. MSE / SSE between input and output image sequences.

            TODO: Use normalized loss as in the original repo
        """
        rec_loss = config['training']['reconstruction_loss']
        N, T = target.shape[0], target.shape[1]
        if rec_loss in ['vqvae', 'VQVAE']:
            # loss = F.mse_loss(prediction, target) / torch.var(target)
            loss = F.mse_loss(prediction, target, reduction='sum') / (torch.var(target) * 2 * N *T)
        elif rec_loss in ['mse', 'MSE']:
            loss = F.mse_loss(input=prediction, target=target, reduction='mean') * 0.5
        elif rec_loss in ['sse', 'SSE']:
            loss = F.mse_loss(input=prediction, target=target, reduction='sum') * 0.5
            loss = loss / (N * T)
        else:
            raise NotImplementedError("Unknown reconstruction loss function.")
        return loss * config['training']['reconstruction_loss_scale']

    def p_c_loss(self,
                 keypoint_coordinates: torch.Tensor,
                 feature_map_sequence: torch.Tensor,
                 image_sequence: torch.Tensor,
                 config: dict) -> torch.Tensor:
        scale = config['training']['pixelwise_contrastive_scale']
        return pc_loss(
            keypoint_coordinates,
            image_sequence,
            feature_map_sequence,
            time_window=config['training']['pixelwise_contrastive_time_window'],
            alpha=config['training']['pixelwise_contrastive_alpha'],
            verbose=False
        ) * scale

    def l1_activation_penalty(self, feature_maps: torch.Tensor, config: dict) -> torch.Tensor:
        feature_map_mean = torch.mean(feature_maps, dim=[-2, -1])
        penalty = torch.mean(torch.abs(feature_map_mean))
        return config['training']['feature_map_regularization'] * penalty

    def step(self, sample: torch.Tensor, target: torch.Tensor, global_epoch_number: int, save_grad_flow_plot: bool,
             save_val_sample: bool, config: dict, mode: str) -> torch.Tensor:

        assert mode in ['training', 'validation']

        if mode == 'training':
            self.optim.zero_grad()

        N, T, C, H, W = sample.shape

        first_frame_fmaps = self.model._appearance_encoder(sample[:, 0, ...])  # (N, C', H', W')
        # first_frame_fmaps = self.model._pre_vq_conv(first_frame_fmaps)  # (N, K, H', W')
        first_frame_vq_loss, first_frame_quantized, first_frame_perplexity, _ = self.model._vq_vae(first_frame_fmaps)
        # TODO: ULOSD paper says to not use the appearence fmap here
        first_frame_kpts = self.model._fmap2kpt(F.softplus(first_frame_fmaps))
        first_frame_gmaps = self.model._kpt2gmap(first_frame_kpts)
        _, first_frame_quantized_gmaps, _, _ = self.model._vq_vae(first_frame_gmaps)
        first_frame_rec = self.model._decoder(
            torch.cat([first_frame_quantized, first_frame_quantized_gmaps, first_frame_quantized], dim=1))

        # Iterate over subsequent frame pairs
        reconstructions = [first_frame_rec.unsqueeze(1)]
        keypoints = [first_frame_kpts.unsqueeze(1)]
        gaussian_maps = [first_frame_gmaps.unsqueeze(1)]
        transported_quantized_maps = []
        source_quantized_maps = []
        feature_maps = [first_frame_fmaps.unsqueeze(1)]
        vq_losses = [first_frame_vq_loss.view(-1, 1)]
        transported_vq_losses = []
        perplexities = [first_frame_perplexity.view(-1, 1)]
        transported_perplexity = []
        for t in range(1, T):
            source_fmap = self.model._encoder(sample[:, t-1, ...])
            source_fmap = self.model._pre_vq_conv(source_fmap)

            target_fmap = self.model._encoder(sample[:, t, ...])
            target_fmap = self.model._pre_vq_conv(target_fmap)
            feature_maps.append(target_fmap.unsqueeze(1))

            # Key-points from source and target frame
            source_kpts = self.model._fmap2kpt(F.softplus(source_fmap))
            target_kpts = self.model._fmap2kpt(F.softplus(target_fmap))
            keypoints.append(target_kpts.unsqueeze(1))

            # Quantize source feature map
            source_vq_loss, source_quantized, source_perplexity, _ = self.model._vq_vae(source_fmap)
            source_quantized_maps.append(source_quantized.unsqueeze(1))

            # Gaussian maps from source and target frames
            source_gmap = self.model._kpt2gmap(source_kpts)
            target_gmap = self.model._kpt2gmap(target_kpts)
            gaussian_maps.append(target_gmap.unsqueeze(1))

            # Get transported feature map
            transported_map = self.model.transport(
                source_gaussian_maps=source_gmap,
                target_gaussian_maps=target_gmap,
                source_feature_maps=source_fmap,
                target_feature_maps=target_fmap
            )

            # Quantize transported map
            transported_vq_loss, transported_quantized, transported_perplexity, _ = self.model._vq_vae(transported_map)
            transported_quantized_maps.append(transported_quantized.unsqueeze(1))

            vq_losses.append((source_vq_loss + transported_vq_loss).view(-1, 1))
            perplexities.append((source_perplexity + transported_perplexity).view(-1, 1))

            # Reconstruct source frame from quantized source feature map
            # and target frame from quantized target feature map
            # TODO: Stack with the first frame feature map and gaussian map, as in ULOSD paper
            source_stacked = torch.cat([first_frame_quantized, first_frame_quantized_gmaps, source_quantized],
                                       dim=1)
            transported_stacked = torch.cat([first_frame_quantized, first_frame_quantized_gmaps, transported_quantized],
                                       dim=1)
            #source_rec = self.model._decoder(source_quantized)
            #target_rec = self.model._decoder(transported_quantized)
            target_rec = self.model._decoder(transported_stacked)
            reconstructions.append(target_rec.unsqueeze(1))

        reconstruction = torch.cat(reconstructions, dim=1)
        keypoint_coordinates = torch.cat(keypoints, dim=1)
        feature_maps = torch.cat(feature_maps, dim=1)
        gaussian_maps = torch.cat(gaussian_maps, dim=1)
        source_quantized_maps = torch.cat(source_quantized_maps, dim=1)
        transported_quantized_maps = torch.cat(transported_quantized_maps, dim=1)
        vq_loss = torch.cat(vq_losses, dim=1).sum() * config['training']['vq_loss_scale']
        perplexity = torch.cat(perplexities, dim=1).mean()

        L_rec = self.loss_func(prediction=reconstruction, target=sample, config=config)

        L_sep = temporal_separation_loss(cfg=config, coords=keypoint_coordinates) \
                * config['training']['separation_loss_scale']

        L_sparse = self.l1_activation_penalty(feature_maps=feature_maps, config=config)

        """
        L_pc = self.p_c_loss(keypoint_coordinates=keypoint_coordinates,
                             feature_map_sequence=gaussian_maps,
                             image_sequence=sample,
                             config=config)
        """

        L_pc = torch.tensor([0.0]).to(self.device)
        L_total = L_rec + vq_loss + L_pc + L_sep + L_sparse

        if mode == 'validation' and config['validation']['save_video'] and save_val_sample:
            # NOTE: This part seems to cause a linear increase in CPU memory usage
            #       Maybe the videos should be saved to the hard-drive instead

            with torch.no_grad():
                torch_img_series_tensor = gen_eval_imgs(sample=sample,
                                                        reconstruction=reconstruction.detach().clamp(-0.5, 0.5),
                                                        key_points=keypoint_coordinates)

                self.writer.add_video(tag='val/reconstruction_sample',
                                      vid_tensor=torch_img_series_tensor,
                                      global_step=global_epoch_number)
                del torch_img_series_tensor

                rgba_frames = torch.empty(size=(1, T, 4, H, W)).to(gaussian_maps.device)
                rgba_frames[:, :, :3, ...] = sample[0:1, ...]
                for k in range(gaussian_maps.shape[2]):
                    interpol_map = F.interpolate(gaussian_maps[0, :, k:k+1, ...], size=(H, W))
                    rgba_frames[:, :, 3] += interpol_map.squeeze()

                self.writer.add_video(tag='val/sample_with_gaussian_maps',
                                      vid_tensor=rgba_frames,
                                      global_step=global_epoch_number)

                del rgba_frames


                """
                self.writer.add_video(tag='val/gaussian_maps',
                                      vid_tensor=torch.cat([sample[0:1, ...],
                                                           torch.sum(upsampled_gaussian_maps,
                                                                     dim=1,
                                                                     keepdim=True).unsqueeze(0)],
                                                           dim=2),
                                      global_step=global_epoch_number)
                """
                self.writer.add_video(tag='val_features/gaussian_maps',
                                      vid_tensor=torch.sum(gaussian_maps[0:1, ...], dim=2).unsqueeze(2),
                                      global_step=global_epoch_number)

                self.writer.add_video(tag='val_features/feature_maps_n0_t0',
                                      vid_tensor=feature_maps[0:1, 0:1, ...].transpose(1, 2),
                                      global_step=global_epoch_number)

                self.writer.add_video(tag='val_features/quantized_transported_maps_n0_t0',
                                      vid_tensor=transported_quantized_maps[0:1, 0:1, ...].transpose(1, 2),
                                      global_step=global_epoch_number)

                self.writer.add_video(tag='val_features/quantized_source_maps_n0_t0',
                                      vid_tensor=source_quantized_maps[0:1, 0:1, ...].transpose(1, 2),
                                      global_step=global_epoch_number)

                self.writer.flush()


        # Log values and backprop. during training
        if mode == 'training':

            self.total_loss_per_iter.append(L_total.item())
            self.rec_loss_per_iter.append(L_rec.item())
            self.sep_loss_per_iter.append(L_sep.item())
            self.sparsity_loss_per_iter.append(L_sparse.item())
            # self.rec_vq_loss_per_iter.append(L_rec_vq.item())
            # self.rec_g_loss_per_iter.append(L_rec_g.item())
            self.pc_loss_per_iter.append(L_pc.item())
            self.vq_loss_per_iter.append(vq_loss.item())
            self.perplexity_per_iter.append(perplexity.item())

            L_total.backward()

            # Clip gradient norm
            nn.utils.clip_grad_norm_(self.model.parameters(), config['training']['clip_norm'])

            with torch.no_grad():
                if save_grad_flow_plot:
                    plot_grad_flow(named_parameters=self.model._encoder.named_parameters(),
                                   epoch=global_epoch_number,
                                   summary_writer=self.writer,
                                   tag_name='encoder')
                    plot_grad_flow(named_parameters=self.model._decoder.named_parameters(),
                                   epoch=global_epoch_number,
                                   summary_writer=self.writer,
                                   tag_name='decoder')
                    plot_grad_flow(named_parameters=self.model._pre_vq_conv.named_parameters(),
                                   epoch=global_epoch_number,
                                   summary_writer=self.writer,
                                   tag_name='pre_vq_conv')
                    plot_grad_flow(named_parameters=self.model._kpt2gmap.named_parameters(),
                                   epoch=global_epoch_number,
                                   summary_writer=self.writer,
                                   tag_name='kpt2gmap')
                    self.writer.flush()

            self.optim.step()
        
        return L_total
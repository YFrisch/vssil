import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchsummary import summary

from src.models.transporter import Transporter
from src.utils.viz import gen_eval_imgs
from src.utils.grad_flow import plot_grad_flow
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

        # assert x.shape[1] == 2
        sample_frame = x[:, 0, ...]
        target_frame = x[:, -1, ...]
        return sample_frame, target_frame

    def loss_func(self, prediction: torch.Tensor, target: torch.Tensor, config: dict) -> torch.Tensor:
        if config['training']['loss_function'] in ['mse', 'l2', 'MSE']:
            return F.mse_loss(input=prediction, target=target)
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

        if mode == 'training':
            self.optim.zero_grad()

        reconstruction = self.model(sample, target).clip(0.0, 1.0)
        loss = self.loss_func(prediction=reconstruction, target=target, config=config)

        if mode == 'training':
            loss.backward()

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
                key_point_maps = self.model.keypointer(sample)[0, ...].unsqueeze(0)
                key_points = self.model._keypoint_loc_mean(features=key_point_maps)
                img = gen_eval_imgs(sample[0, ...].unsqueeze(0),
                                    target[0, ...].unsqueeze(0),
                                    reconstruction[0, ...].unsqueeze(0),
                                    key_points=key_points).squeeze(0)

                self.writer.add_image(tag='val/example', img_tensor=img, global_step=global_epoch_number)

        return loss


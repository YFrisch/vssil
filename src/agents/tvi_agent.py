import torch
import torch.nn as nn
from torch.utils.data import Dataset

from .abstract_agent import AbstractAgent
from ..models.lstm import CustomLSTM


class TVI_Agent(AbstractAgent):

    def __init__(self,
                 dataset: Dataset = None,
                 config: dict = None):

        assert config is not None, "No config given for the TVI agent!"

        super(TVI_Agent, self).__init__(
            name="Temporal Variational Inference Agent",
            dataset=dataset,
            config=config
        )

        in_size = self.data_set.get_sample_shape()[-1]

        """ q(sigma|tau) predicts sequences of options
            given sequences of trajectories. 
            
            The NN parameterization yields three fc heads,
            one predicting the binary termination variable b_t at every time-step,
            and one predicting the mean and one the variance
            of each latent z_i at each time-step
        
        """
        self.q = CustomLSTM(
            input_size=in_size,
            num_out_classes=[2, config['model']['num_latent_vars'], config['model']['num_latent_vars']],
            lstm_hidden_size=config['model']['q']['hidden_size'],
            num_lstm_layers=config['model']['q']['num_layers'],
            num_fc_layers=[1, 1, 1],
            num_heads=config['model']['q']['num_heads'],
            fc_activation=nn.ReLU(),
            out_activations=[nn.Softmax, nn.Identity, nn.Softplus],
            seq_len=self.data_set.timesteps_per_sample
        )

        """ pi(a|s,a,sigma) predicts the mean and variance of the
            (continuous) low-level actions (joint velocities)
        """
        self.pi = CustomLSTM(
            input_size=in_size,
            num_out_classes=[config['model']['num_latent_vars'], config['model']['num_latent_vars']],
            lstm_hidden_size=config['model']['pi']['hidden_size'],
            num_lstm_layers=config['model']['pi']['num_layers'],
            num_heads=config['model']['pi']['num_heads'],
            num_fc_layers=[1, 1],
            fc_activation=nn.ReLU(),
            out_activations=[nn.Identity, nn.Softplus],
            seq_len=self.data_set.timesteps_per_sample
        )

        """ eta(sigma|s, a, sigma) predicts the binary termination variables and the 
            latent variables given a history of states, actions and latents
        """
        self.eta = CustomLSTM(
            input_size=in_size,
            num_out_classes=[2, config['model']['num_latent_vars'], config['model']['num_latent_vars']],
            lstm_hidden_size=config['model']['eta']['hidden_size'],
            num_lstm_layers=config['model']['eta']['num_layers'],
            num_heads=config['model']['eta']['num_heads'],
            num_fc_layers=[1, 1, 1],
            fc_activation=nn.ReLU(),
            out_activations=[nn.Softmax, nn.Identity, nn.Softplus],
            seq_len=self.data_set.timesteps_per_sample
        )

        self.optim = torch.optim.Adam(
            params=[*self.q.parameters(), *self.pi.parameters(), *self.eta.parameters()],
            lr=config['training']['lr']
        )

    def sample_pi(self, x: torch.Tensor):
        """ Sample action from pi network parameterizing the mean and variance
            of the action distribution
        """
        _y = self.pi(x)
        return _y

    def loss_func(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pass

    def train_step(self, sample: torch.Tensor, target: torch.Tensor, config: dict) -> torch.Tensor:
        pass

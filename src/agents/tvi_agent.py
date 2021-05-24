import torch

from .abstract_agent import AbstractAgent
from ..models.lstm import CustomLSTM


class TVI_Agent(AbstractAgent):

    def __init__(self,
                 config: dict = None):

        assert config is not None, "No config given for the TVI agent!"

        super(TVI_Agent, self).__init__()
        self.q = CustomLSTM()
        self.pi = CustomLSTM()
        self.eta = CustomLSTM()

    def loss_func(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pass

    def train(self, config: dict = None):
        pass

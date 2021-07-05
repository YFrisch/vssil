import torch
from torch.utils.data import Dataset

from .abstract_agent import AbstractAgent
from src.models.ulosd import ULOSD

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

        # Each key-point is represented by an individual feature map
        self.n_feature_maps = config['model']['n_feature_maps']

        self.model = ULOSD(
            input_width=160,
            input_height=60,
            config=config
        )
        self.optim = ...

    def loss_func(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pass

    def train_step(self, sample: torch.Tensor, target: torch.Tensor, config: dict) -> torch.Tensor:
        pass
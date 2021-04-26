""" Abstract class for agents to inherit from.
    Already implements common methods.
"""

import torch


class AbstractAgent:

    def __init__(self,
                 name: str = "AbstractAgent",
                 config: dict = None):

        self.name = name
        self.setup(config)

        self.model = None
        self.train_data_loader = None
        self.eval_data_loader = None
        self.val_data_loader = None
        self.optim = None
        self.scheduler = None

    def setup(self, config: dict = None):
        raise NotImplementedError

    def loss_func(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def preprocess(self, x: torch.Tensor, config: dict) -> torch.Tensor:
        """ Preprocess samples. """
        return x

    def train(self, config: dict = None):

        for epoch in config['epochs']:

            self.model.train()

            for i, (sample, target) in enumerate(self.train_data):

                # Preprocessing
                sample = self.preprocess(sample)

                # Forward pass
                prediction = self.model(sample)

                # Calc loss
                L = self.loss_func(prediction, target)

                # Backprop
                L.backward()

                # Optim. step
                self.optim.step()
                self.scheduler.step()

    def validate(self, config: dict = None):
        raise NotImplementedError

    def evaluate(self, config = None):
        raise NotImplementedError

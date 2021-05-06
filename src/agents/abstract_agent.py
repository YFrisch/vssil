""" Abstract class for agents to inherit from.
    Already implements common methods.
"""
import os
import shutil
import time

import yaml
from pathlib import Path
from datetime import datetime, date

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np


class AbstractAgent:

    def __init__(self,
                 name: str = "Abstract Agent",
                 dataset: Dataset = None,
                 config: dict = None):

        self.name = name

        self.data_set = dataset

        assert not (dataset.timesteps_per_sample == -1 and config['training']['batch_size'] > 1), \
            "Batch size > 1 is not yet supported for whole trajectories."

        self.train_data_loader = DataLoader(dataset,
                                            batch_size=config['training']['batch_size'],
                                            shuffle=True)

        self.eval_data_loader = DataLoader(dataset,
                                           batch_size=config['training']['batch_size'],
                                           shuffle=True)

        self.val_data_loader = DataLoader(dataset,
                                          batch_size=config['training']['batch_size'],
                                          shuffle=True)

        self.device = config['device'] if config['device'] is not None else "cpu"

        self.log_dir = None

        self.writer = None

        self.best_val_loss = None

        self.is_setup = False

        # The following parts are implemented by inheriting classes
        self.model = None
        self.optim = None
        self.scheduler = None  # TODO: Not used yet

    def setup(self, config: dict = None):
        """ Sets up all relevant attributes for logging results. """

        print(f"##### Setting up {self.name} on {self.device}.")

        year, month, day, hour, minute = datetime.now().year, datetime.now().month, datetime.now().day, \
                                         datetime.now().hour, datetime.now().minute

        self.log_dir = config['log_dir'] + f"{year}_{month}_{day}_{hour}_{minute}/"
        if os.path.exists(self.log_dir) and os.path.isdir(self.log_dir):
            shutil.rmtree(self.log_dir)

        os.makedirs(os.path.join(self.log_dir, 'checkpoints/'), exist_ok=True)

        with open(self.log_dir + "config.yml", "w") as config_file:
            yaml.dump(config, config_file)

        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.is_setup = True

    def loss_func(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def preprocess(self, x: torch.Tensor, config: dict) -> torch.Tensor:
        """ Preprocess samples. """
        return x

    def load_checkpoint(self, chckpt_path: str = None):
        """ Load state dict for internal model from given path."""

        assert self.model is not None, "Model not initialized."
        assert chckpt_path is not None, "No checkpoint path given."

        self.model.load_state_dict(torch.load(chckpt_path))

    def make_train_val_split(self, config: dict = None):
        # TODO: Make this exhaustive for x-validation (Make sure each sample is used)
        val_len = int(config['validation']['val_split'] * len(self.data_set))
        train_len = len(self.data_set) - val_len

        train_set, val_set = random_split(self.data_set, [train_len, val_len])

        self.train_data_loader = DataLoader(train_set,
                                            batch_size=config['training']['batch_size'],
                                            shuffle=True,
                                            num_workers=0)

        self.val_data_loader = DataLoader(val_set,
                                          batch_size=config['training']['batch_size'],
                                          shuffle=True,
                                          num_workers=0)

    def train(self, config: dict = None):
        raise NotImplementedError

    def validate(self, training_epoch: int, config: dict = None):

        print("##### Validating:")
        time.sleep(1)

        self.model.eval()

        loss_per_sample = []

        with torch.no_grad():

            for i, sample in enumerate(tqdm(self.val_data_loader)):

                sample, target = self.preprocess(sample, config)  # Sample is in (N, T, C, H, W)
                sample, target = sample.to(self.device), target.to(self.device)

                predictions = None

                for t in range(sample.shape[1]):
                    frame_prediction = self.model(sample[:, t, ...])
                    if predictions is None:
                        predictions = frame_prediction
                    else:
                        predictions = torch.cat([predictions, frame_prediction], dim=1)

                sample_loss = self.loss_func(predictions.squeeze(), target.squeeze())
                loss_per_sample.append(sample_loss.cpu().numpy())

        avg_loss = np.mean(loss_per_sample)
        self.writer.add_scalar(tag="val/loss", scalar_value=avg_loss, global_step=training_epoch)
        print("##### Average loss:", avg_loss)
        print("\n")
        time.sleep(0.1)

        if self.best_val_loss is None:
            self.best_val_loss = avg_loss
            torch.save(self.model.state_dict(),
                       self.log_dir + f'checkpoints/chckpt_e{training_epoch}.PTH')
        elif avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            # Save current model
            torch.save(self.model.state_dict(),
                       self.log_dir + f'checkpoints/chckpt_e{training_epoch}.PTH')
        else:
            pass

        return loss_per_sample

    def evaluate(self, config: dict = None):

        print("##### Evaluating:")
        time.sleep(0.1)

        self.model.eval()

        # TODO: Load model from config['evaluation']['chckpt_path']

        loss_per_sample = []

        with torch.no_grad():
            for i, sample in enumerate(tqdm(self.eval_data_loader)):
                sample, target = self.preprocess(sample, config)
                sample, target = sample.to(self.device), target.to(self.device)

                prediction = self.model(sample)

                sample_loss = self.loss_func(prediction, target)
                loss_per_sample.append(sample_loss.cpu().numpy())

        print("##### Average loss:", np.mean(loss_per_sample))
        time.sleep(0.1)

        return loss_per_sample

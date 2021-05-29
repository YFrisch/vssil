""" Abstract class for agents to inherit from.
    Already implements common methods.
"""
import os
import shutil
import time
import gc

import yaml
from pathlib import Path
from datetime import datetime, date

import torch
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import KFold


class AbstractAgent:

    def __init__(self,
                 name: str = "Abstract Agent",
                 dataset: Dataset = None,
                 config: dict = None):

        self.name = name

        self.data_set = dataset
        self.kfold = None
        self.train_data_loader = None
        self.val_data_loader = None
        self.eval_data_loader = None

        assert not (dataset.timesteps_per_sample == -1 and config['training']['batch_size'] > 1), \
            "Batch size > 1 is not supported for whole trajectories."

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
        """ Sets up all relevant attributes for training and logging results. """

        print(f"##### Setting up {self.name} on {self.device}.")

        year, month, day, hour, minute = datetime.now().year, datetime.now().month, datetime.now().day, \
                                         datetime.now().hour, datetime.now().minute

        self.log_dir = os.path.join(os.getcwd(), config['log_dir'] + f"/{year}_{month}_{day}_{hour}_{minute}/")
        if os.path.exists(self.log_dir) and os.path.isdir(self.log_dir):
            shutil.rmtree(self.log_dir)

        os.makedirs(os.path.join(self.log_dir, 'checkpoints/'), exist_ok=True)

        with open(self.log_dir + "config.yml", "w") as config_file:
            yaml.dump(config, config_file)

        self.writer = SummaryWriter(log_dir=self.log_dir)

        self.kfold = KFold(n_splits=config['training']['k_folds'], shuffle=True)

        self.is_setup = True

    def loss_func(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def preprocess(self, x: torch.Tensor, config: dict) -> (torch.Tensor, (torch.Tensor, torch.Tensor)):
        """ Pre-process samples. """
        return x

    def load_checkpoint(self, chckpt_path: str = None):
        """ Load state dict for internal model from given path."""

        assert self.model is not None, "Model not initialized."
        assert chckpt_path is not None, "No checkpoint path given."

        self.model.load_state_dict(torch.load(chckpt_path))

    def train_step(self, sample: torch.Tensor, target: torch.Tensor, config: dict) -> torch.Tensor:
        raise NotImplementedError

    def train(self, config: dict = None):
        """ General training loop. """

        self.setup(config=config)

        assert self.is_setup, "Model was not set up."

        print(f"##### Training {self.name}.")
        time.sleep(0.01)

        # Iterate over k-folds
        for fold, (train_ids, val_ids) in enumerate(self.kfold.split(self.data_set)):

            print(f"##### Fold {fold}:")
            time.sleep(0.01)

            # Define training and evaluation data
            self.train_data_loader = DataLoader(
                dataset=self.data_set,
                batch_size=config['training']['batch_size'],
                sampler=SubsetRandomSampler(train_ids)
            )
            self.val_data_loader = DataLoader(
                dataset=self.data_set,
                batch_size=config['training']['batch_size'],
                sampler=SubsetRandomSampler(val_ids)
            )

            # Iterate over epochs
            for epoch in range(0, config['training']['epochs']):

                print(f"##### Epoch {epoch}:")
                time.sleep(0.01)

                self.model.train()

                loss_per_iter = []

                # Iterate over samples
                for i, sample in enumerate(tqdm(self.train_data_loader)):
                    with torch.no_grad():
                        sample, target = self.preprocess(sample, config)  # (N, T, C, H, W)

                    sample, target = sample.to(self.device), target.to(self.device)

                    loss = self.train_step(sample, target, config)
                    assert loss is not None, "No loss returned during training."
                    loss_per_iter.append(loss.detach().cpu().numpy())

                    del sample, target, loss
                    gc.collect()

                avg_loss = np.mean(loss_per_iter)
                print(f"\nEpoch: {epoch}|{config['training']['epochs']}\t\t Avg. loss: {avg_loss}\n")
                self.writer.add_scalar(tag="train/loss", scalar_value=avg_loss, global_step=epoch)

                # Validate
                if not epoch % config['validation']['freq']:
                    self.validate(training_epoch=epoch, config=config)

    def validate(self, training_epoch: int, config: dict = None):

        print("##### Validating:")
        time.sleep(0.1)

        self.model.eval()

        loss_per_sample = []

        with torch.no_grad():

            for i, sample in enumerate(tqdm(self.val_data_loader)):
                sample, target = self.preprocess(sample, config)  # Sample is in (N, T, C, H, W)
                sample, target = sample.to(self.device), target.to(self.device)

                prediction = self.model(sample)

                sample_loss = self.loss_func(prediction.squeeze(), target.squeeze())
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

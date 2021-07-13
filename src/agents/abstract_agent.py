""" Abstract class for agents to inherit from.
    Already implements common methods.
"""
import os
import sys
import shutil
import time
import gc

import json
import yaml
from pathlib import Path
from datetime import datetime, date

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split, SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.model_selection import KFold

from src.utils.json import pretty_json


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

        assert not (config['data']['timesteps_per_sample'] == -1 and config['training']['batch_size'] > 1), \
            "Batch size > 1 is not supported for whole trajectories."

        self.device = config['device'] if config['device'] is not None else "cpu"

        self.log_dir = None

        self.writer = None

        self.best_val_loss = None

        self.is_setup = False

        # Logged values, losses, metrics, etc.
        self.loss_per_iter = []

        # The following parts are implemented by inheriting classes
        self.model = None
        self.optim = None
        self.scheduler = None  # TODO: Not used yet

    def setup(self, config: dict = None):
        """ Sets up all relevant attributes for training and logging results. """

        if torch.cuda.device_count() > 1:
            print(f"##### Setting up {self.name} on {torch.cuda.device_count()} gpus.")
            self.model = torch.nn.DataParallel(self.model)
        else:
            print(f"##### Setting up {self.name} on {self.device}.")

        self.log_dir = os.path.join(os.getcwd(), config['log_dir'])
        # self.log_dir = os.path.join(os.getcwd(), config['log_dir'] + f"/{year}_{month}_{day}_{hour}_{minute}/")
        if os.path.exists(self.log_dir) and os.path.isdir(self.log_dir):
            shutil.rmtree(self.log_dir)

        os.makedirs(os.path.join(self.log_dir, 'checkpoints/'), exist_ok=True)

        with open(self.log_dir + "config.yml", "w") as config_file:
            yaml.dump(config, config_file)

        self.writer = SummaryWriter(log_dir=self.log_dir)
        # Add graph of model
        # self.writer.add_graph(self.model)
        # Add config dict
        # self.writer.add_hparams(config, {})
        self.writer.add_text("parameters", pretty_json(config))
        self.writer.flush()

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

    def reset_logged_values(self):
        """ Resets all logged values (metrics, losses, ...). """
        self.loss_per_iter = []

    def log_values(self, fold: int, epoch: int, epochs_per_fold: int):
        global_epoch = fold*epochs_per_fold + epoch
        avg_loss = np.mean(self.loss_per_iter)
        self.writer.add_scalar(tag="train/loss", scalar_value=avg_loss, global_step=global_epoch)

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

            # Define training and evaluation data
            self.train_data_loader = DataLoader(
                dataset=self.data_set,
                batch_size=config['training']['batch_size'],
                sampler=SubsetRandomSampler(train_ids),
                num_workers=config['data']['num_workers'],
                pin_memory=True
            )
            self.val_data_loader = DataLoader(
                dataset=self.data_set,
                batch_size=config['training']['batch_size'],
                sampler=SubsetRandomSampler(val_ids),
                num_workers=config['data']['num_workers'],
                pin_memory=True
            )

            # Iterate over epochs
            epochs_per_fold = config['training']['epochs']
            for epoch in range(epochs_per_fold):

                print(f"##### Fold {fold} Epoch {epoch}:")
                time.sleep(0.01)

                self.model.train()

                self.reset_logged_values()

                # # Iterate over samples
                # for i, sample in enumerate(tqdm(self.train_data_loader)):

                # Iterate over steps
                for i in tqdm(range(config['training']['steps_per_epoch'])):

                    sample = next(iter(self.train_data_loader))

                    with torch.no_grad():
                        sample, target = self.preprocess(sample, config)  # (N, T, C, H, W)

                    sample, target = sample.to(self.device), target.to(self.device)

                    loss = self.train_step(sample, target, config)
                    assert loss is not None, "No loss returned during training."
                    self.loss_per_iter.append(loss.detach().cpu().numpy())

                    del sample, target, loss
                    gc.collect()

                avg_loss = np.mean(self.loss_per_iter)
                print(f"\nEpoch: {epoch}|{config['training']['epochs']}\t\t Avg. loss: {avg_loss}\n")

                self.log_values(fold=fold, epoch=epoch, epochs_per_fold=epochs_per_fold)

                # Validate
                if not epoch % config['validation']['freq']:
                    self.validate(training_fold=fold, training_epoch=epoch, config=config)

                sys.stdout.flush()

    def validation_loss(self,
                        sample: torch.Tensor,
                        prediction: torch.Tensor,
                        target: torch.Tensor,
                        config: dict) -> torch.Tensor:
        """ Returns the loss to use for validation.
            Note that this can differ from self.loss_function()!
        """
        return self.loss_func(prediction=prediction, target=target)

    def validate(self, training_fold: int, training_epoch: int, config: dict = None):

        print("##### Validating:")
        time.sleep(0.1)

        self.model.eval()

        loss_per_sample = []

        with torch.no_grad():

            for i, sample in enumerate(tqdm(self.val_data_loader)):
                sample, target = self.preprocess(sample, config)  # Sample is in (N, T, C, H, W)
                sample, target = sample.to(self.device), target.to(self.device)

                prediction = self.model(sample)

                sample_loss = self.validation_loss(sample=sample,
                                                   prediction=prediction.squeeze(),
                                                   target=target.squeeze(),
                                                   config=config)

                loss_per_sample.append(sample_loss.cpu().numpy())

        avg_loss = np.mean(loss_per_sample)
        epochs_per_fold = config['training']['epochs']
        global_epoch = training_fold * epochs_per_fold + training_epoch
        self.writer.add_scalar(tag="val/loss", scalar_value=avg_loss, global_step=global_epoch)
        print("##### Average loss:", avg_loss)
        print("\n")
        time.sleep(0.1)

        if self.best_val_loss is None:
            self.best_val_loss = avg_loss
            torch.save(self.model.state_dict(),
                       self.log_dir + f'checkpoints/chckpt_f{training_fold}_e{training_epoch}.PTH')
        elif avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            # Save current model
            torch.save(self.model.state_dict(),
                       self.log_dir + f'checkpoints/chckpt_f{training_fold}_e{training_epoch}.PTH')
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

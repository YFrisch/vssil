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

        self.uses_multiple_gpus = False

        self.is_setup = False

        # Logged values, losses, metrics, etc.
        self.loss_per_iter = []

        # The following parts are implemented by inheriting classes
        self.model = None
        self.optim = None
        self.scheduler = None

    def warm_start(self, config: dict):
        """ Loads a model checkpoint specified in the config dict. """
        print(f"##### Loading checkpoint {config['warm_start_checkpoint']}.")
        self.load_checkpoint(config['warm_start_checkpoint'])

    def setup(self, config: dict = None):
        """ Sets up all relevant attributes for training and logging results. """

        if torch.cuda.device_count() > 1 and config['multi_gpu']:
            print(f"##### Setting up {self.name} on {torch.cuda.device_count()} gpus.")
        else:
            print(f"##### Setting up {self.name} on {self.device}.")

        if torch.cuda.is_available():
            config['used_gpus'] = str([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])

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
        weight_names = [name for name, param in self.model.named_parameters()]
        self.writer.add_text("weights", '\n'.join(weight_names))
        self.writer.flush()

        if config['training']['k_folds'] > 1:
            self.kfold = KFold(n_splits=config['training']['k_folds'], shuffle=True)

        self.reset_optim_and_scheduler(config=config)

        if config['warm_start'] is True:
            self.warm_start(config=config)

        self.is_setup = True

    def loss_func(self,
                  prediction: torch.Tensor,
                  target: torch.Tensor,
                  config: dict) -> torch.Tensor:
        raise NotImplementedError

    def preprocess(self,
                   x: torch.Tensor,
                   config: dict) -> (torch.Tensor, (torch.Tensor, torch.Tensor)):
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

    def log_values(self,
                   fold: int,
                   epoch: int,
                   epochs_per_fold: int):
        global_epoch = fold * epochs_per_fold + epoch
        avg_loss = np.mean(self.loss_per_iter)
        self.writer.add_scalar(tag="train/loss", scalar_value=avg_loss, global_step=global_epoch)

    def reset_optim_and_scheduler(self, config: dict):

        if config['training']['optim'] in ['Adam', 'adam', 'ADAM']:
            self.optim = torch.optim.Adam(
                params=self.model.parameters(),
                lr=config['training']['initial_lr'],
                weight_decay=config['training']['l2_weight_decay']
            )
        else:
            raise NotImplementedError("Optimizer not implemented yet.")

        if config['training']['lr_scheduler'] in ['CosineAnnealingLR']:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=self.optim,
                T_max=config['training']['epochs'],
                eta_min=config['training']['min_lr']
            )
        elif config['training']['lr_scheduler'] in ['StepLR']:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=self.optim,
                step_size=config['training']['lr_scheduler_epoch_steps'],
                gamma=0.5
            )
        elif config['training']['lr_scheduler'] in [None, 'None', 'NONE', 'none']:
            self.scheduler = None
        else:
            raise NotImplementedError("LR Scheduler not implemented yet.")

    def step(self,
             sample: torch.Tensor,
             target: torch.Tensor,
             global_epoch_number: int,
             save_grad_flow_plot: bool,
             save_val_sample: bool,
             config: dict,
             mode: str) -> torch.Tensor:
        """ One step of forwarding the model input and calculating the loss."""
        raise NotImplementedError

    def train(self, config: dict = None):
        """ General training loop. """

        self.setup(config=config)

        assert self.is_setup, "Model was not set up."

        print(f"##### Training {self.name}.")
        time.sleep(0.01)

        # Iterate over k-folds
        if config['training']['k_folds'] > 1:
            folds = enumerate(self.kfold.split(self.data_set))

        else:
            # Use 10% of the data for validation, if no fold for x-validation is given
            train_ids = range(0, int(len(self.data_set) * 0.9))
            val_ids = range(int(len(self.data_set) * 0.9) + 1, len(self.data_set))
            folds = [(0, (train_ids, val_ids))]

        for fold, (train_ids, val_ids) in folds:

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

                    if i == 0:
                        save_grad_flow_plot = True
                    else:
                        save_grad_flow_plot = False

                    loss = self.step(sample,
                                     target,
                                     global_epoch_number=fold * epochs_per_fold + epoch,
                                     save_grad_flow_plot=save_grad_flow_plot,
                                     save_val_sample=False,
                                     config=config,
                                     mode='training')
                    assert loss is not None, "No loss returned during training."
                    self.loss_per_iter.append(loss.detach().cpu().numpy())

                    del sample, target, loss
                    gc.collect()

                avg_loss = np.mean(self.loss_per_iter)
                print(f"\nEpoch: {epoch}|{config['training']['epochs']}\t\t Avg. loss: {avg_loss}\n")

                self.log_values(fold=fold, epoch=epoch, epochs_per_fold=epochs_per_fold)
                for param_group in self.optim.param_groups:
                    self.writer.add_scalar(tag="train/lr",
                                           scalar_value=param_group['lr'],
                                           global_step=fold * epochs_per_fold + epoch)

                # Validate
                if not epoch % config['validation']['freq']:
                    with torch.no_grad():
                        self.validate(training_fold=fold, training_epoch=epoch, config=config)

                sys.stdout.flush()

                if self.scheduler is not None:
                    self.scheduler.step()

            # Reset scheduler for each fold
            self.reset_optim_and_scheduler(config)

    def validate(self,
                 training_fold: int,
                 training_epoch: int,
                 config: dict = None):

        print("##### Validating:")
        time.sleep(0.1)

        self.model.eval()

        loss_per_sample = []

        epochs_per_fold = config['training']['epochs']
        global_epoch = training_fold * epochs_per_fold + training_epoch

        for i, sample in enumerate(tqdm(self.val_data_loader)):
            sample, target = self.preprocess(sample, config)  # Sample is in (N, T, C, H, W)
            sample, target = sample.to(self.device), target.to(self.device)

            if i == 0:
                save_val_sample = True
            else:
                save_val_sample = False

            sample_loss = self.step(sample,
                                    target,
                                    global_epoch_number=global_epoch,
                                    save_grad_flow_plot=False,
                                    save_val_sample=save_val_sample,
                                    config=config,
                                    mode='validation')

            loss_per_sample.append(sample_loss.cpu().numpy())

        avg_loss = np.mean(loss_per_sample)
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

                sample_loss = self.loss_func(prediction, target, config=config)
                loss_per_sample.append(sample_loss.cpu().numpy())

        print("##### Average loss:", np.mean(loss_per_sample))
        time.sleep(0.1)

        return loss_per_sample

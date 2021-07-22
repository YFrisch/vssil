import os
from os.path import isfile, join

import torch
import torch.nn as nn
from torchvision.models.utils import load_state_dict_from_url

from .inception3 import CustomInception3


def get_img_coordinates(h: int, w: int, normalize: bool):
    """ TODO """
    x_range = torch.arange(w, dtype=torch.float32)
    y_range = torch.arange(h, dtype=torch.float32)
    if normalize:
        x_range = (x_range / (w - 1)) * 2 - 1
        y_range = (y_range / (h - 1)) * 2 - 1
    image_x = x_range.unsqueeze(0).repeat_interleave(h, 0)
    image_y = y_range.unsqueeze(0).repeat_interleave(w, 0).t()
    return image_x, image_y


def partial_load_state_dict(model: torch.nn.Module, loaded_dict: torch.ParameterDict):
    """ Loads all named parameters of the given model
        from the given loaded state_dict.

    :param model: The model whose parameters are to update
    :param loaded_dict: The state dict to update from
    """
    model_state_dict = model.state_dict()

    for name, param in loaded_dict.items():
        if name not in model_state_dict:
            continue
        if isinstance(param, torch.nn.Parameter):
            param = param.data
        model_state_dict[name].copy_(param)


def load_inception_weights(inception_net: CustomInception3, config: dict):
    """ Loads inception net weight (pretrained on ImageNet) from file or url. """
    if not isfile(join(config['log_dir'], "inception.pth")):
        # Create dir if it does not exist
        os.makedirs(name=config['log_dir'], exist_ok=True)
        # Load state dict from url
        state_dict = load_state_dict_from_url(config['model']['inception_url'])
        torch.save(state_dict, f=join(config['log_dir'], "inception.pth"))
    else:
        state_dict = torch.load(join(config['log_dir'], "inception.pth"))

    # Only get the relevant parts of the state dict needed for the custom module
    partial_load_state_dict(inception_net, state_dict)


activation_dict = {
    'relu': nn.ReLU(),
    'ReLU': nn.ReLU(),
    'RELU': nn.ReLU(),
    'ELU': nn.ELU(),
    'elu': nn.ELU(),
    'LeakyRELU': nn.LeakyReLU(),
    'LeakyReLU': nn.LeakyReLU(),
    'PReLU': nn.PReLU(),
    'prelu': nn.PReLU()
}

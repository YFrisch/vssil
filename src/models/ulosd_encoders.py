import torch.nn as nn

from .layers import Conv2DSamePadding
from .utils import activation_dict


def make_encoder(input_shape: tuple, config: dict):
    """ Image encoder.

        Iteratively halving the input width and doubling
        the number of channels, until the width of the
        feature maps is reached.

    """
    encoder_module_list = []
    # Adjusted input shape (for add_coord_channels)
    encoder_input_shape = (input_shape[1] + 2, *input_shape[2:])
    assert len(encoder_input_shape) == 3

    # First, expand the input to an initial number of filters
    # num_channels = encoder_input_shape[0]
    num_channels = 3
    encoder_module_list.append(
        Conv2DSamePadding(
            in_channels=num_channels,
            out_channels=config['model']['n_init_filters'],
            kernel_size=(config['model']['conv_kernel_size'], config['model']['conv_kernel_size']),
            stride=(1, 1),
            activation=config['model']['encoder_hidden_activations']
        )
    )
    encoder_module_list.append(
        nn.BatchNorm2d(num_features=config['model']['n_init_filters'])
    )
    input_width = encoder_input_shape[-1]
    num_channels = config['model']['n_init_filters']
    # Apply additional layers
    for _ in range(config['model']['n_convolutions_per_res']):
        encoder_module_list.append(
            Conv2DSamePadding(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=(config['model']['conv_kernel_size'], config['model']['conv_kernel_size']),
                stride=(1, 1),
                activation=config['model']['encoder_hidden_activations']
            )
        )
        encoder_module_list.append(
            nn.BatchNorm2d(num_features=num_channels)
        )

    while True:
        # Reduce resolution
        encoder_module_list.append(
            Conv2DSamePadding(
                in_channels=num_channels,
                out_channels=num_channels * 2,
                kernel_size=(config['model']['conv_kernel_size'], config['model']['conv_kernel_size']),
                stride=(2, 2),
                activation=config['model']['encoder_hidden_activations']
            )
        )
        encoder_module_list.append(
            nn.BatchNorm2d(num_features=num_channels * 2)
        )

        # Apply additional layers
        for _ in range(config['model']['n_convolutions_per_res']):
            encoder_module_list.append(
                Conv2DSamePadding(
                    in_channels=num_channels * 2,
                    out_channels=num_channels * 2,
                    kernel_size=(config['model']['conv_kernel_size'], config['model']['conv_kernel_size']),
                    stride=(1, 1),
                    activation=config['model']['encoder_hidden_activations']
                )
            )
            encoder_module_list.append(
                nn.BatchNorm2d(num_features=num_channels * 2)
            )
        input_width //= 2
        num_channels *= 2
        if input_width <= config['model']['feature_map_width']:
            break

    # Final layer that maps to the desired number of feature_maps
    encoder_module_list.append(
        Conv2DSamePadding(
            in_channels=num_channels,
            out_channels=config['model']['n_feature_maps'],
            kernel_size=(config['model']['conv_kernel_size'], config['model']['conv_kernel_size']),
            stride=(1, 1),
            activation=nn.Softplus()
        )
    )
    return nn.Sequential(*encoder_module_list), encoder_input_shape


def make_appearance_encoder(input_shape: tuple, config: dict):
    """ """
    appearance_module_list = []
    # Adjusted input shape (for add_coord_channels)
    encoder_input_shape = (input_shape[1] + 2, *input_shape[2:])
    assert len(encoder_input_shape) == 3

    # First, expand the input to an initial number of filters
    # num_channels = encoder_input_shape[0]
    num_channels = 3
    appearance_module_list.append(
        Conv2DSamePadding(
            in_channels=num_channels,
            out_channels=config['model']['n_init_filters'],
            kernel_size=(config['model']['conv_kernel_size'], config['model']['conv_kernel_size']),
            stride=(1, 1),
            activation=config['model']['encoder_hidden_activations']
        )
    )
    appearance_module_list.append(
        nn.BatchNorm2d(num_features=config['model']['n_init_filters'])
    )
    input_width = encoder_input_shape[-1]
    num_channels = config['model']['n_init_filters']
    # Apply additional layers
    for _ in range(config['model']['n_convolutions_per_res']):
        appearance_module_list.append(
            Conv2DSamePadding(
                in_channels=num_channels,
                out_channels=num_channels,
                kernel_size=(config['model']['conv_kernel_size'], config['model']['conv_kernel_size']),
                stride=(1, 1),
                activation=config['model']['encoder_hidden_activations']
            )
        )
        appearance_module_list.append(
            nn.BatchNorm2d(num_features=num_channels)
        )

    while True:
        # Reduce resolution
        appearance_module_list.append(
            Conv2DSamePadding(
                in_channels=num_channels,
                out_channels=num_channels * 2,
                kernel_size=(config['model']['conv_kernel_size'], config['model']['conv_kernel_size']),
                stride=(2, 2),
                activation=config['model']['encoder_hidden_activations']
            )
        )
        appearance_module_list.append(
            nn.BatchNorm2d(num_features=num_channels * 2)
        )
        # Apply additional layers
        for _ in range(config['model']['n_convolutions_per_res']):
            appearance_module_list.append(
                Conv2DSamePadding(
                    in_channels=num_channels * 2,
                    out_channels=num_channels * 2,
                    kernel_size=(config['model']['conv_kernel_size'], config['model']['conv_kernel_size']),
                    stride=(1, 1),
                    activation=config['model']['encoder_hidden_activations']
                )
            )
            appearance_module_list.append(
                nn.BatchNorm2d(num_features=num_channels * 2)
            )
        input_width //= 2
        num_channels *= 2
        if input_width <= config['model']['feature_map_width']:
            break

    # Final layer that maps to the desired number of feature_maps
    appearance_module_list.append(
        Conv2DSamePadding(
            in_channels=num_channels,
            out_channels=config['model']['n_feature_maps'],
            kernel_size=(config['model']['conv_kernel_size'], config['model']['conv_kernel_size']),
            stride=(1, 1),
            activation=nn.Softplus()
        )
    )

    return nn.Sequential(*appearance_module_list)

import torch.nn as nn
import numpy as np

from .layers import Conv2DSamePadding


def make_decoder(encoder_input_shape: tuple, config: dict):
    decoder_module_list = []
    num_channels = config['model']['n_feature_maps'] * 3 + 2
    # num_levels = np.log2(init_input_width / config['model']['feature_map_width'])
    num_levels = np.log2(encoder_input_shape[1] / config['model']['feature_map_width'])
    if num_levels % 1:
        raise ValueError(f"The input image width must be a two potency"
                         f" of the feature map width, but got {encoder_input_shape[1]}"
                         f" and {config['model']['feature_map_width']}!")

    # Iteratively double the resolution by upsampling
    for _ in range(int(num_levels)):

        num_out_channels = num_channels//2

        decoder_module_list.append(
            nn.Upsample(
                scale_factor=(2.0, 2.0),
                mode='bilinear',
                align_corners=True
            )
        )

        decoder_module_list.append(
            Conv2DSamePadding(
                in_channels=num_channels,
                out_channels=num_out_channels,
                kernel_size=(config['model']['conv_kernel_size'], config['model']['conv_kernel_size']),
                stride=(1, 1),
                activation=nn.ELU(alpha=1)
            )
        )

        for _ in range(config['model']['n_convolutions_per_res'] - 1):
            decoder_module_list.append(
                Conv2DSamePadding(
                    in_channels=num_out_channels,
                    out_channels=num_out_channels,
                    kernel_size=(config['model']['conv_kernel_size'], config['model']['conv_kernel_size']),
                    stride=(1, 1),
                    activation=nn.ELU(alpha=1)
                )
            )

        num_channels //= 2

    # Adjust channels
    decoder_module_list.append(
        Conv2DSamePadding(
            in_channels=num_out_channels,
            out_channels=3,
            kernel_size=(1, 1),
            stride=(1, 1),
            activation=nn.Identity()
        )
    )

    return nn.Sequential(*decoder_module_list)

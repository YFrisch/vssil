import torch.nn as nn
import numpy as np

from .layers import Conv2DSamePadding


def make_decoder(encoder_input_shape: tuple, n_encoder_output_channels: int, config: dict):

    decoder_module_list = []

    # Number of channels from the combined representation
    # (First frame gaussian maps, first frame feature maps, current frame gaussian maps, coord channels)
    n_decoder_input_channels = \
        (config['model']['n_feature_maps'] + n_encoder_output_channels + config['model']['n_feature_maps'] + 2)

    decoder_input_shape = \
        (n_encoder_output_channels, config['model']['feature_map_height'], config['model']['feature_map_width'])

    # Conv layer to adjust the decoder input channels
    decoder_module_list.append(
        Conv2DSamePadding(
            in_channels=n_decoder_input_channels,
            out_channels=n_encoder_output_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            activation=config['model']['decoder_hidden_activations']
        )
    )

    num_channels = decoder_input_shape[0]
    num_levels = np.log2(encoder_input_shape[-1] / decoder_input_shape[-1])
    if num_levels % 1:
        raise ValueError(f"The input image width must be a two potency"
                         f" of the feature map width, but got {encoder_input_shape[1]}"
                         f" and {config['model']['feature_map_width']}!")

    for _ in range(int(num_levels)):

        # Upsampling layer, doubling the resolution
        decoder_module_list.append(
            nn.Upsample(
                scale_factor=(2.0, 2.0),
                mode='bilinear',
                #mode='bicubic',
                align_corners=True
            )
        )

        num_in_channels = num_channels
        num_out_channels = num_channels // 2
        first = True

        for _ in range(config['model']['n_convolutions_per_res']):

            decoder_module_list.append(
                Conv2DSamePadding(
                    in_channels=num_in_channels,
                    out_channels=num_out_channels,
                    kernel_size=(config['model']['conv_kernel_size'], config['model']['conv_kernel_size']),
                    stride=(1, 1),
                    activation=config['model']['decoder_hidden_activations']
                )
            )
            decoder_module_list.append(
                nn.BatchNorm2d(num_features=num_out_channels)
            )
            if first:
                num_in_channels //= 2
                first = False

        num_channels //= 2

    # Adjust channels
    decoder_module_list.append(
        Conv2DSamePadding(
            in_channels=num_channels,
            out_channels=3,
            kernel_size=(1, 1),
            stride=(1, 1),
            activation=nn.Identity()
        )
    )

    return nn.Sequential(*decoder_module_list)

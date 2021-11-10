"""

    PyTorch implementation of https://github.com/google-research/google-research/blob/master/video_structure/losses.py

"""

import torch


def temporal_separation_loss(cfg: dict, coords: torch.Tensor) -> torch.Tensor:
    """ Encourages key-point to have different temporal trajectories.

    :param cfg: Configuration dictionary
    :param coords: Key-point coordinates tensor in (N, T, C, 3)
    :return: The separation loss
    """

    # Trajectories are centered first
    x_coordinates = coords[..., 0] - torch.mean(coords[..., 0], dim=1, keepdim=True)  # (N, T, C)
    y_coordinates = coords[..., 1] - torch.mean(coords[..., 1], dim=1, keepdim=True)

    # Compute the pair-wise distance matrix
    x_1 = x_coordinates.unsqueeze(-1)  # (N, T, C, 1)
    x_2 = x_coordinates.unsqueeze(-2)  # (N, T, 1, C)
    y_1 = y_coordinates.unsqueeze(-1)
    y_2 = y_coordinates.unsqueeze(-2)
    d = ((x_1 - x_2)**2 + (y_1 - y_2)**2)  # (N, T, C, C)

    # Average across time
    d = torch.mean(d, dim=1)  # (N, 1, C, C)

    # Transform by gaussian
    loss_matrix = torch.exp(-d / (2.0 * cfg['training']['separation_loss_sigma']**2))
    loss_matrix = torch.mean(loss_matrix, dim=0)  # Average across batch

    loss = torch.sum(loss_matrix)

    # Substract values on diagonal (1 per key-point)
    loss = loss - cfg['model']['n_feature_maps']

    # Normalize to [0, 1]
    loss = loss / (cfg['model']['n_feature_maps'] * (cfg['model']['n_feature_maps'] - 1))

    return loss

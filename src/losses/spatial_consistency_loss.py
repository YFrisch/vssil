import numpy as np
import torch


def spatial_consistency_loss(coords: torch.Tensor, cfg: dict) -> torch.Tensor:
    """ Encourages key-points to have small - or more consistent - spatial movement
        across time-steps

    :param coords: Torch tensor of key-point coordinates in (N, T, C, 2/3)
    :param cfg: Additional configuration dictionary
    :return: The consistency loss
    """

    assert coords.dim() == 4

    N = coords.shape[0]

    alpha = cfg['model']['feature_map_gauss_sigma']/2.0
    gamma = 0.9

    distance_loss = 0
    vel_loss = 0

    """
    for t in np.arange(0, coords.shape[1] - 1):
        diff = torch.norm(coords[:, t + 1, ...] - coords[:, t, ...], dim=2)
        # Average across batch and kp
        avg_diff = torch.mean(diff)
        thresholded_diff = max(0, abs(avg_diff - alpha))
        loss = loss + thresholded_diff
    """
    for t in np.arange(1, coords.shape[1]):
        # NOTE: The next line also includes the differences in key-point intensity (3rd axis)
        distance_to_first_frame = torch.norm(coords[:, t, ...] - coords[:, 0, ...], dim=2)
        # Average across batch
        # TODO: Sum over key-points instead of averaging?
        avg_distance = torch.mean(distance_to_first_frame)
        thresholded_diff = max(0, abs(avg_distance) - alpha)
        # TODO: Check this!
        distance_loss = distance_loss + gamma**(t-1) * thresholded_diff

    for t in np.arange(2, coords.shape[1]):
        distance_between_last_frames = torch.norm(coords[:, t-2, ...] - coords[:, t-1, ...], dim=2)
        distance_to_last_frame = torch.norm(coords[:, t-1, ...] - coords[:, t, ...], dim=2)
        velocity_diff = torch.abs(distance_between_last_frames - distance_to_last_frame)
        avg_diff = torch.mean(velocity_diff)
        thresholded_diff = max(0, avg_diff - 0.1)
        vel_loss = vel_loss + thresholded_diff

    return torch.Tensor([distance_loss + vel_loss]).to(coords.device)



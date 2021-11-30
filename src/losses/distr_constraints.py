import torch
import torch.nn as nn

from .sinkhorn_diff import SinkhornDistance


def wasserstein_constraint(kpt_sequence: torch.Tensor) -> torch.Tensor:
    """ Penalizes a high EMD between gaussians with centers at the key-points' coordinates.

    :param kpt_sequence: Sequence of key-point positions in (N, T, K, D)
    """

    dist = SinkhornDistance(eps=0.1, max_iter=100, reduction='mean', device=kpt_sequence.device)

    N, T, K, D = kpt_sequence.shape

    costs = []
    for t in range(T - 1):

        cost, pi, C = dist(kpt_sequence[:, t, ...], kpt_sequence[:, t+1, ...])
        costs.append(cost)

    costs = torch.stack([*costs]).unsqueeze(1)  # (T-1, 1)

    return costs





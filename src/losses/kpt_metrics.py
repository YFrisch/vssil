""" This script contains metrics to evaluate key-point trajectories at inference time."""
import torch
import numpy as np
from kornia.filters import canny
from kornia.feature import TFeat, MKDDescriptor, SIFTFeature

from src.utils.kpt_utils import get_box_within_image_border
from .spatial_consistency_loss import spatial_consistency_loss


def combined_metric(image_sequence: torch.Tensor,
                    kpt_sequence: torch.Tensor,
                    method: str = 'norm',
                    time_window: int = 3):
    """ Combining the spatial consistency metric and the patchwise contrastive metric."""
    return spatial_consistency_loss(keypoint_coordinates=kpt_sequence) + \
        patchwise_contrastive_metric(image_sequence, kpt_sequence, method, time_window)


def patchwise_contrastive_metric(image_sequence: torch.Tensor,
                                 kpt_sequence: torch.Tensor,
                                 method: str = 'norm',
                                 time_window: int = 3,
                                 patch_size: tuple = (7, 7),
                                 alpha: float = 0.1):
    """ Contrasts pixel patches around key-points.
        Positive examples are drawn from the same key-point at time-steps in the given time-window.
        Negative examples are drawn from other key-points at any time-step
            or the same key-point outside of the time-window.

    :param image_sequence: Tensor of sequential images in (N, T, C, H, W)
    :param kpt_sequence: Tensor of key-point coordinates in (N, T, K, D)
    :param method: Method to use:
        'mean': Compares the mean patch differences
        'norm': Compares the image norm of the patch differences
        'vssil': Uses the pixelwise-contrastive feature representations
        'tfeat': Uses tfeat encodings to compare the image patches
    :param time_window: Window size of positive examples around current the current time-step
        E.g. time_window=3 uses t-1 and t+1 as positives for t
        At t=0 and t=T, the window size is reduced.
    :param patch_size: Size of the patch so extract from the input, around the key-point
        If these would extend the image borders, they are moved to within the borders.
        TODO: Fix with padding instead ?
    :param alpha: Allowance for pos / neg similarity
    """

    N, T, C, H, W = image_sequence.shape
    assert kpt_sequence.shape[0] == N, "images and kpts dont share batch size dim"
    assert kpt_sequence.shape[1] == T, "images and kpts dont share time dim"
    _, _, K, D = kpt_sequence.shape

    # To reduce the computational effort, the extracted patches are saved and re-used by demand
    patch_sequence = torch.empty(size=(N, T, K, C, patch_size[0], patch_size[1]))
    evaluated_kpts = []

    L = torch.empty(size=(N, T, K)).to(kpt_sequence.device)

    # Iterate over time-steps
    for t in range(T):

        # Iterate over key-points
        for k in range(K):

            #
            #   ANCHOR
            #

            if (t, k) in evaluated_kpts:
                anchor_patch = patch_sequence[:, t, k, ...].float()
            else:
                x_min, x_max, y_min, y_max = get_box_within_image_border(kpt_sequence, patch_size, H, W, t, k)
                anchor_patch = image_sequence[:, t, :, x_min: x_max + 1, y_min: y_max + 1].float()
                patch_sequence[:, t, k, ...] = anchor_patch
                evaluated_kpts.append((t, k))

            #
            #   POSITIVES
            #

            L_pos = torch.tensor([0]).to(kpt_sequence.device)
            t_range = np.arange(max(0, t - int(time_window/2)), min(T - 1, t + int(time_window/2)) + 1)
            # t_range = np.arange(0, T)
            for t_p in t_range:
                if t_p == t:
                    continue
                if (t_p, k) in evaluated_kpts:
                    positive_patch = patch_sequence[:, t_p, k, ...].float()
                else:
                    x_min, x_max, y_min, y_max = get_box_within_image_border(kpt_sequence, patch_size, H, W, t_p, k)
                    positive_patch = image_sequence[:, t_p, :, x_min: x_max + 1, y_min: y_max + 1].float()
                    patch_sequence[:, t_p, k, ...] = positive_patch
                    evaluated_kpts.append((t_p, k))

                L_pos = L_pos + torch.norm(positive_patch - anchor_patch, p=2)
                L_pos = L_pos + torch.norm(kpt_sequence[:, t, k, :] - kpt_sequence[:, t_p, k, :], p=2)

            L_pos = (L_pos / (len(t_range) - 1)) if len(t_range) > 2 else L_pos

            #
            #   NEGATIVES
            #

            L_neg = torch.tensor([0]).to(kpt_sequence.device)
            # for t_n in range(0, T):
            for t_n in t_range:
                for k_n in range(0, K):
                    if (t_n in t_range or t_n == t) and k_n == k:
                        continue
                    else:
                        if (t_n, k_n) in evaluated_kpts:
                            negative_patch = patch_sequence[:, t_n, k_n].float()
                        else:
                            x_min, x_max, y_min, y_max = get_box_within_image_border(kpt_sequence, patch_size, H, W,
                                                                                     t_n, k_n)
                            negative_patch = image_sequence[:, t_n, :, x_min:x_max + 1, y_min:y_max + 1].float()
                            patch_sequence[:, t_n, k_n, ...] = negative_patch
                            evaluated_kpts.append((t_n, k_n))

                    L_neg = L_neg + torch.norm(negative_patch - anchor_patch, p=2)
                    L_neg = L_neg + torch.norm(kpt_sequence[:, t, k, :] - kpt_sequence[:, t_n, k_n, :], p=2)

            L_neg = L_neg / (T*(K - 1) + T - len(t_range) + 1)
            print(f't: {t} k: {k} = ', max(L_pos - L_neg + alpha, torch.tensor([0.0])).mean().item())
            L[:, t, k] = max(L_pos - L_neg + alpha, torch.tensor([0.0]))

    return torch.mean(L, dim=[0, 2])


def grad_tracking_metric(patch_sequence: torch.Tensor) -> torch.Tensor:
    """ Evaluates how consistent the gradients of patches around key-points are
        across time.

        Should be as low as possible.

    :param patch_sequence: Tensor of image patches in (N, T, K, C, H', W')
    :return: Metric, averaged across the batch and key-points
    """

    N, T, K, C, Hp, Wp = patch_sequence.shape

    # Stack batch, time and key-point dimension
    patch_sequence = patch_sequence.view((N*T*K, C, Hp, Wp))

    # TODO: Sobel filter
    mags, grads = canny(input=patch_sequence, low_threshold=0.2, high_threshold=0.5, kernel_size=(3, 3))  # (N*T*K, 1, H', W')

    # Unstack dimensions
    grads = grads.view((N, T, K, Hp, Wp))
    mags = mags.view((N, T, K, Hp, Wp))

    # Empty tensor for the sum distances
    grad_dists = torch.empty((N, T - 1, K))
    mag_dists = torch.empty((N, T - 1, K))

    for t in range(T - 1):
        grad_dists[:, t, :] = torch.abs(torch.mean(grads[:, t, ...], dim=(-2, -1)) -
                                        torch.mean(grads[:, t + 1, ...], dim=(-2, -1)))
        mag_dists[:, t, :] = torch.abs(torch.mean(mags[:, t, ...], dim=(-2, -1)) -
                                       torch.mean(mags[:, t + 1, ...], dim=(-2, -1)))

    # Sum across time
    grad_dists = torch.sum(grad_dists, dim=1)  # (N, K)
    mag_dists = torch.sum(mag_dists, dim=1)  # (N, K)

    # Average sum across batch and key-points
    return torch.mean(0.5*grad_dists + 0.5*mag_dists)


def ft_tracking_metric(patch_sequence: torch.Tensor) -> torch.Tensor:
    """ Evaluates how consistent the features of patches around key-points are
        across time.

        Should be as low as possible.

    :param patch_sequence: Tensor of image patches in (N, T, K, C, H', W')
    :return: Metric, averaged across the batch and key-points
    """

    N, T, K, C, Hp, Wp = patch_sequence.shape

    d = 32

    assert Hp == Wp, "Use squared patches"

    desc = MKDDescriptor(patch_size=Hp, output_dims=d)

    # Stack batch, time and key-point dimension
    patch_sequence = patch_sequence.view((N*T*K*C, 1, Hp, Wp))

    # Empty tensor for the sum distances
    ft_dists = torch.empty((N * (T - 1) * K * C, 1))

    for n in range(N):
        for t in range(T - 1):
            for k in range(K):
                for c in range(C):
                    ft_dists[n * t * k * c] = torch.norm(
                        desc(patch_sequence[n * t * k * c].unsqueeze(0)) -
                            desc(patch_sequence[n * (t + 1) * k * c].unsqueeze(0)))

    ft_dists = ft_dists.view((N, T-1, K, C))

    # Sum across time and channels
    ft_dists = torch.sum(ft_dists, dim=(1, 3))  # (N, K)

    # Average across batch and key-points
    return torch.mean(ft_dists)


def visual_difference_metric(patch_sequence: torch.Tensor) -> torch.Tensor:
    """ Evaluates the perceptual differences of the
        image patches around key-points.

        Should be as high as possible.

    :param patch_sequence: Tensor of image patches in (N, T, K, C, H', W')
    :return: Metric
    """

    N, T, K, C, Hp, Wp = patch_sequence.shape

    d = 32

    assert Hp == Wp, "Use squared patches"

    desc = MKDDescriptor(patch_size=Hp, output_dims=d)

    # Stack batch, time, kpt and channel dimensions
    patch_sequence = patch_sequence.view((N*T*K*C, 1, Hp, Wp))

    # Extract features
    fts = desc(patch_sequence)

    # Tensor holding perceptual differences between key-points
    _fts = fts.view((N * T * C, K, 1, d))
    __fts = fts.view((N * T * C, 1, K, d))
    fts_dist_tensor = torch.norm(_fts - __fts, p=2, dim=-1)**2
    fts_dist_tensor = fts_dist_tensor.view((N, T, C, K, K))

    # Sum over key-points (Should be as high as possible)
    fts_dist_tensor = torch.sum(fts_dist_tensor, dim=[-2, -1])/(K*(K-1))

    # Average across batch, time-steps and channel
    return torch.mean(fts_dist_tensor)


def visual_similarity_metric(patch_sequence: torch.Tensor) -> torch.Tensor:
    """ Evaluates the perceptual differences of the
        image patches around a key-point over time.

        Should be as low as possible.

    :param patch_sequence: Tensor of image patches in (N, T, K, C, H', W')
    :return: Metric
    """

    N, T, K, C, Hp, Wp = patch_sequence.shape

    d = 32

    assert Hp == Wp, "Use squared patches"

    desc = MKDDescriptor(patch_size=Hp, output_dims=d)

    # Stack batch, time, kpt and channel dimensions
    patch_sequence = patch_sequence.view((N*T*K*C, 1, Hp, Wp))

    # Extract features
    fts = desc(patch_sequence)

    # Tensor holding perceptual differences between key-points
    _fts = fts.view((N * K * C, T, 1, d))
    __fts = fts.view((N * K * C, 1, T, d))
    fts_dist_tensor = torch.norm(_fts - __fts, p=2, dim=-1)**2
    fts_dist_tensor = fts_dist_tensor.view((N, K, C, T, T))

    # Sum over time (Should be as low as possible)
    fts_dist_tensor = torch.sum(fts_dist_tensor, dim=[-2, -1])/(T*(T-1))

    # Average across batch, key-points and channel
    return torch.mean(fts_dist_tensor)

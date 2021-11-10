""" This script contains metrics to evaluate key-point trajectories at inference time."""
import torch
import numpy as np

from .spatial_consistency_loss import spatial_consistency_loss

def get_box_within_image_border(kpt_sequence, patch_size, H, W, t, k):

    x = (kpt_sequence[:, t, k, 0] + 1) / 2 * W
    y = (-kpt_sequence[:, t, k, 1] + 1) / 2 * H

    x_min, x_max = max(0, x - int(patch_size[0] / 2)), min(W - 1, x + int(patch_size[0] / 2))
    y_min, y_max = max(0, y - int(patch_size[1] / 2)), min(H - 1, y + int(patch_size[1] / 2))

    while int(x_max) - int(x_min) >= patch_size[0]:
        x_max -= 1
    while int(y_max) - int(y_min) >= patch_size[1]:
        y_max -= 1

    #print(f'x: {x}\t{int(x_min)} - {int(x_max)}')
    #print(f'y: {y}\t{int(y_min)} - {int(y_max)}')
    #print()

    return int(x_min), int(x_max), int(y_min), int(y_max)


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
                                 patch_size: tuple = (3, 3),
                                 alpha: float = 1.0):
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
        TODO: Fix with padding instead.
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

    # Iterate over key-points
    for k in range(K):

        # Iterate over time-steps
        for t in range(T):

            #
            #   ANCHOR
            #

            if (t, k) in evaluated_kpts:
                anchor_patch = patch_sequence[:, t, k, ...].float()
            else:
                x_min, x_max, y_min, y_max = get_box_within_image_border(kpt_sequence, patch_size, H, W, t, k)
                anchor_patch = image_sequence[:, t, :, x_min:x_max+1, y_min:y_max+1].float()
                patch_sequence[:, t, k, ...] = anchor_patch
                evaluated_kpts.append((t, k))

            #
            #   POSITIVES
            #

            L_pos = torch.tensor([0]).to(kpt_sequence.device)
            t_range = np.arange(max(0, t - int(time_window/2)), min(T - 1, t + int(time_window/2)) + 1)
            for t_p in t_range:
                if t_p == t:
                    continue
                if (t_p, k) in evaluated_kpts:
                    positive_patch = patch_sequence[:, t_p, k, ...].float()
                else:
                    x_min, x_max, y_min, y_max = get_box_within_image_border(kpt_sequence, patch_size, H, W, t_p, k)
                    positive_patch = image_sequence[:, t_p, :, x_min:x_max + 1, y_min:y_max + 1].float()
                    patch_sequence[:, t_p, k, ...] = positive_patch
                    evaluated_kpts.append((t_p, k))

                # TODO: Make modular for different methods
                L_pos = L_pos + torch.norm(positive_patch - anchor_patch, p=2)
            L_pos = (L_pos / (len(t_range)-1)) if len(t_range) > 2 else L_pos

            #
            #   NEGATIVES
            #

            L_neg = torch.tensor([0]).to(kpt_sequence.device)
            for t_n in range(0, T):
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

                    # TODO: Make modular for different methods
                    L_neg = L_pos + torch.norm(negative_patch - anchor_patch, p=2)
            L_neg = L_neg / (T*(K-1) + T-len(t_range)+1)
            L[:, t, k] = max(L_pos - L_neg + alpha, torch.tensor([0.0]))

    L = torch.sum(L, dim=2)
    return torch.mean(L)

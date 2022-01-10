import numpy as np
import torch


def get_box_within_image_border(kpt_sequence, patch_size, H, W, t, k):

    w = (kpt_sequence[:, t, k, 1] + 1) / 2 * W
    h = (-kpt_sequence[:, t, k, 0] + 1) / 2 * H

    h_min, h_max = max(0, h - int(patch_size[0] / 2)), min(H - 1, h + int(patch_size[0] / 2))
    w_min, w_max = max(0, w - int(patch_size[1] / 2)), min(W - 1, w + int(patch_size[1] / 2))

    while int(h_max) - int(h_min) >= patch_size[0]:
        h_max -= 1
    while int(w_max) - int(w_min) >= patch_size[1]:
        w_max -= 1

    return int(h_min), int(h_max), int(w_min), int(w_max)


def get_image_patches(image_sequence: torch.Tensor,
                      kpt_sequence: torch.Tensor,
                      patch_size: tuple):
    """ Returns the image patches from image_sequence of size patch_size around kpt_sequence

        Todo: Check case of N > 1

        (Non-differentiable)
    """
    assert image_sequence.shape[:2] == kpt_sequence.shape[:2]
    N, T, C, H, W = image_sequence.shape
    _, _, K, D = kpt_sequence.shape

    patch_sequence = torch.empty(size=(N, T, K, C, patch_size[0], patch_size[1]))

    for n in range(N):
        for t in range(T):
            for k in range(K):
                h_min, h_max, w_min, w_max = get_box_within_image_border(kpt_sequence, patch_size, H, W, t, k)
                patch_sequence[n, t, k, ...] = image_sequence[n, t, :, h_min:h_max + 1, w_min:w_max + 1]

    return patch_sequence

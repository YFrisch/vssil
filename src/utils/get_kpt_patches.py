import numpy as np
import torch


def get_box_within_image_border(kpt_sequence, patch_size, H, W, t, k):

    """ Returns patch coordinates from key-points,
        taking into account the image dimensions.
    """

    x = (kpt_sequence[:, t, k, 0] + 1) / 2 * W
    y = (-kpt_sequence[:, t, k, 1] + 1) / 2 * H

    x_min, x_max = max(0, x - int(patch_size[0] / 2)), min(W, x + int(patch_size[0] / 2))
    y_min, y_max = max(0, y - int(patch_size[1] / 2)), min(H, y + int(patch_size[1] / 2))
    x_min, x_max = int(x_min), int(np.floor(x_max))
    y_min, y_max = int(y_min), int(np.floor(y_max))

    while (x_max - x_min) < patch_size[0]:

        if x_max == W:
            x_min -= 1
        elif x_min == 0:
            x_max += 1
        else:
            x_max += 1

    while (y_max - y_min) < patch_size[1]:
        if y_max == H:
            y_min -= 1
        elif y_min == 0:
            y_max += 1
        else:
            y_max += 1

    return int(x_min), int(x_max), int(y_min), int(y_max)


def get_kpt_patches(img_sequence: torch.Tensor,
                    kpt_sequence: torch.Tensor,
                    patch_size: tuple) -> torch.Tensor:
    """ Returns the patches of the image sequence around the key-points positions.

        The patches are extracted in a non-differentiable way
        by converting the key-points to discrete tensor indices.

    :param img_sequence: Tensor of image frames in (N, T, C, H, W)
    :param kpt_sequence: Tensor of key-point encodings in (N, T, K, D)
    :param patch_size: Shape of patch in (x,y) to extract around each key-point
    """

    assert img_sequence.shape[:2] == kpt_sequence.shape[:2], "Batch sizes or time-steps don't match."

    N, T, C, H, W = img_sequence.shape

    _, _, K, D = kpt_sequence.shape

    # Tensor holding the patches
    patch_sequence = torch.empty((N, T, K, C, patch_size[0], patch_size[1]))

    for t in range(T):
        for k in range(K):
            x_min, x_max, y_min, y_max = get_box_within_image_border(kpt_sequence, patch_size, H, W, t, k)
            patch_sequence[:, t, k, ...] = img_sequence[:, t, :, x_min: x_max, y_min: y_max]

    return patch_sequence

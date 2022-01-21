import numpy as np
import torch


def kpts_2_img_coordinates(kpt_coordinates: torch.Tensor,
                           img_shape: tuple) -> torch.Tensor:
    """ Converts the key-point coordinates from video structure format [-1,1]x[-1,1]
        to image coordinates in [0,H]x[0,W].

    :param kpt_coordinates: Torch tensor in (N, T, K, 2/3)
    :param img_shape: Tuple of image size (H, W)
    """
    _kpt_coordinates = torch.empty_like(kpt_coordinates)
    _kpt_coordinates[..., 0] = (-kpt_coordinates[..., 0] + 1.0) * img_shape[-1]/2.0
    _kpt_coordinates[..., 1] = (kpt_coordinates[..., 1] + 1.0) * img_shape[-2]/2.0
    return _kpt_coordinates


def get_box_within_image_border(kpt_sequence, patch_size, H, W, t, k):
    """ Returns the indices to select a patch of size 'patch_size'
        of on '(H, W)' image around the key-point 'k' at time-step 't'
        from the given key-point sequence.

        If the patch would exceed the image borders, it is moved to inside the image.
    """
    w = (kpt_sequence[:, t, k, 1] + 1) / 2 * W
    h = (-kpt_sequence[:, t, k, 0] + 1) / 2 * H

    h_min, h_max = max(0, h - int(patch_size[0] / 2)), min(H - 1, h + int(patch_size[0] / 2))
    w_min, w_max = max(0, w - int(patch_size[1] / 2)), min(W - 1, w + int(patch_size[1] / 2))
    h_min, h_max = int(np.floor(h_min)), int(np.floor(h_max))
    w_min, w_max = int(np.floor(w_min)), int(np.floor(w_max))

    while (h_max - h_min) < patch_size[0]:
        h_max += 1
        h_max = np.clip(h_max, 0, H)
        if (h_max - h_min) < patch_size[0]:
            h_min -= 1
            h_min = np.clip(h_min, 0, H)

    while (h_max - h_min) > patch_size[0]:
        h_max -= 1
        h_max = np.clip(h_max, 0, H)
        if (h_max - h_min) > patch_size[0]:
            h_min += 1
            h_min = np.clip(h_min, 0, H)

    while (w_max - w_min) < patch_size[1]:
        w_max += 1
        w_max = np.clip(w_max, 0, W)
        if (w_max - w_min) < patch_size[1]:
            w_min -= 1
            w_min = np.clip(w_min, 0, W)

    while (w_max - w_min) > patch_size[1]:
        w_max -= 1
        w_max = np.clip(w_max, 0, W)
        if (w_max - w_min) > patch_size[1]:
            w_min += 1
            w_min = np.clip(w_min, 0, W)

    return h_min, h_max, w_min, w_max


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
                patch = image_sequence[n, t, :, h_min:h_max, w_min:w_max]
                assert patch.shape[-2:] == patch_size
                patch_sequence[n, t, k, ...] = patch

    return patch_sequence

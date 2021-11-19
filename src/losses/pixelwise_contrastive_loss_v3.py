import random

import torch
import torch.nn.functional as F
from kornia.filters import laplacian
from itertools import product


def pixelwise_contrastive_loss_patch_based(
        keypoint_coordinates: torch.Tensor,
        image_sequence: torch.Tensor,
        grid: torch.Tensor,
        pos_range: int = 1,
        patch_size: tuple = (9, 9),
        alpha: float = 0.1
) -> torch.Tensor:

    """ This version of the pixelwise contrastive loss uses torch.grid_sample(...) to extract the (interpolated)
        patches around the key-points.
        The grid is obtained by multiplying a fixed grid, just depending on the input image shape, with the key-points.

    :param keypoint_coordinates: Tensor of key-point coordinates in (N, T, K, 2/3)
    :param image_sequence: Tensor of sequential frames in (N, T, C, H, W)
    :param grid: Tensor of grid positions in (N, H'', W'', 2) where (H'', W'') is the patch size
    :param pos_range: Range of time-steps to consider as matches ([anchor - range, anchor + range])
    :param patch_size: Patch dimensions
    :param alpha: Threshold for matching
    :return:
    """

    #
    #   Checking tensor shapes
    #

    assert keypoint_coordinates.dim() == 4
    assert image_sequence.dim() == 5
    assert keypoint_coordinates.shape[0:2] == image_sequence.shape[0:2]

    N, T, C, H, W = image_sequence.shape
    K, D = keypoint_coordinates.shape[2:4]

    #
    #   Generate the sample grid and sample the patches using it
    #

    unstacked_kpts = keypoint_coordinates.view((N*T*K, D))

    sample_grids = torch.empty_like(grid).to(image_sequence.device)
    for ntk in range(sample_grids.shape[0]):
        sample_grids[ntk, :, :, 0] = grid[ntk, :, :, 0] + unstacked_kpts[ntk, 1]
        sample_grids[ntk, :, :, 1] = grid[ntk, :, :, 1] - unstacked_kpts[ntk, 0]

    # Expand image sequence for key-point dimension
    _image_sequence = image_sequence.unsqueeze(2).repeat((1, 1, K, 1, 1, 1))
    _image_sequence = _image_sequence.view((N*T*K, C, H, W))

    patches = F.grid_sample(
        input=_image_sequence,
        grid=sample_grids,
        mode='bilinear',
        align_corners=False
    ).to(image_sequence.device)

    """
    print(patches.shape)
    fig, ax = plt.subplots(3, 8, figsize=(20, 12))
    for t in range(3):
        for k in range(8):
            ax[t][k].imshow(patches[t*k].detach().cpu().permute(1, 2, 0) + 0.5)
    plt.show()
    exit()
    """

    grads = laplacian(
        input=patches,
        kernel_size=3
    ).to(image_sequence.device)
    grads = grads.view((N, T, K, 3, patch_size[0], patch_size[1]))

    patches = patches.view((N, T, K, 3, patch_size[0], patch_size[1]))

    # Calculate contrastive loss from features
    L = torch.zeros((N,)).to(image_sequence.device)
    for t in range(T):

        for k in range(K):

            #
            #   Anchor feature representation
            #

            # anchor_ft = ft[:, t, k, ...]

            time_steps = range(max(0, t - pos_range), min(T, t + pos_range + 1))

            matches = [(t_i, k) for t_i in time_steps]
            matches.remove((t, k))

            non_matches = list(product(time_steps, range(K)))
            non_matches.remove((t, k))

            for (t_p, k_p) in matches:

                try:
                    non_matches.remove((t_p, k_p))
                except ValueError:
                    continue

            #
            #   Match loss
            #

            L_p = torch.tensor((N,)).to(image_sequence.device)
            # for (t_p, k_p) in positives:
            for (t_p, k_p) in [random.choice(matches)]:
                L_p = L_p + torch.norm(patches[:, t, k, ...] - patches[:, t_p, k_p, ...],
                                       p=2, dim=[1, 2, 3])**2
                L_p = L_p + torch.norm(grads[:, t, k, ...] - grads[:, t_p, k_p, ...],
                                       p=2, dim=[1, 2, 3]) ** 2
                L_p = L_p + torch.norm(keypoint_coordinates[:, t, k, ...] - keypoint_coordinates[:, t_p, k_p, ...],
                                       p=2)**2
            # L_p /= len(positives)

            #
            #   Non-match loss
            #

            L_n = torch.tensor((N,)).to(image_sequence.device)
            # for (t_n, k_n) in negatives:
            for (t_n, k_n) in [random.choice(non_matches)]:
                L_n = L_n + torch.norm(patches[:, t, k, ...] - patches[:, t_n, k_n, ...], p=2, dim=[1, 2, 3])**2
                L_n = L_n + torch.norm(grads[:, t, k, ...] - grads[:, t_n, k_n, ...],
                                       p=2, dim=[1, 2, 3]) ** 2
                L_n = L_n + torch.norm(keypoint_coordinates[:, t, k, ...] - keypoint_coordinates[:, t_n, k_n, ...],
                                       p=2) ** 2
            # L_n /= len(negatives)

            #
            #   Total Loss
            #

            L = torch.add(
                L,
                torch.maximum(L_p - L_n + alpha, torch.zeros((N,)).to(image_sequence.device))
            )

    L = L / (T*K)

    return torch.mean(L)


if __name__ == "__main__":

    N, T, C, H, W = 8, 8, 3, 64, 64

    K = 4

    time_window = 5
    patch_size = (9, 9)

    pos_range = max(int(time_window / 2), 1) if time_window > 1 else 0
    center_index = int(patch_size[0] / 2)

    step_matrix = torch.ones(patch_size + (2,)).to('cuda:0')

    step_w = 2 / W
    step_h = 2 / H

    for k in range(0, patch_size[0]):
        for l in range(0, patch_size[1]):
            step_matrix[k, l, 0] = (l - center_index) * step_w
            step_matrix[k, l, 1] = (k - center_index) * step_h

    grid = step_matrix.unsqueeze(0).repeat((N * T * K, 1, 1, 1)).to('cuda:0')

    fake_img = torch.rand(size=(N, T, C, H, W)).to('cuda:0')

    fake_kpts = torch.rand(size=(N, T, K, 3)).to('cuda:0')
    fake_kpts[..., 2] = 1.0

    print(pixelwise_contrastive_loss_patch_based(
        keypoint_coordinates=fake_kpts,
        image_sequence=fake_img,
        pos_range=pos_range,
        grid=grid,
        patch_size=(9, 9),
        alpha=0.01
    ))


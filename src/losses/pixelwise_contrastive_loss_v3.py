import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import laplacian
from itertools import combinations, product


def pixelwise_contrastive_loss_v3(
        keypoint_coordinates: torch.Tensor,
        image_sequence: torch.Tensor,
        grid: torch.Tensor,
        pos_range: int = 1,
        patch_size: tuple = (9, 9),
        alpha: float = 0.1
) -> torch.Tensor:

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

    grads = laplacian(
        input=patches,
        kernel_size=3
    ).to(image_sequence.device)
    grads = grads.view((N, T, K, 3, patch_size[0], patch_size[1]))

    # TODO: The following two nested loops could be merged

    # Create feature vectors from patches
    ft = torch.empty((N, T, K, 4)).to(image_sequence.device)
    ft[..., :3] = unstacked_kpts.view((N, T, K, -1))
    for n in range(N):
        for t in range(T):
            for k in range(K):
                ft[n, t, k, 3] = torch.sum(
                    grads[n, t, k][(torch.abs(grads[n, t, k, ...]) > 0.1)]
                )

    # Calculate contrastive loss from features
    L = torch.zeros((N,)).to(image_sequence.device)
    for t in range(T):

        for k in range(K):

            #
            #   Anchor feature representation
            #

            anchor_ft = ft[:, t, k, ...]

            positives = [(t_i, k) for t_i in range(
                max(0, t - pos_range), min(T, t + pos_range + 1)
            )]

            positives.remove((t, k))

            negatives = list(product(range(T), range(K)))

            negatives.remove((t, k))
            for (t_p, k_p) in positives:

                try:
                    negatives.remove((t_p, k_p))
                except ValueError:
                    continue

            #
            #   Match loss
            #
            L_p = torch.tensor((N,)).to(image_sequence.device)
            for (t_p, k_p) in positives:
                L_p = L_p + torch.norm(anchor_ft - ft[:, t_p, k_p, ...], p=2, dim=1)
            L_p /= len(positives)

            #
            # Non-match loss
            #
            L_n = torch.tensor((N,)).to(image_sequence.device)
            for (t_n, k_n) in negatives:
                L_n = L_n + torch.norm(anchor_ft - ft[:, t_n, k_n, ...], p=2, dim=1)
            L_p /= len(negatives)

            # TODO: Correct to just average over key-points AND time-steps AND batch?
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

    print(pixelwise_contrastive_loss_v3(
        keypoint_coordinates=fake_kpts,
        image_sequence=fake_img,
        pos_range=pos_range,
        grid=grid,
        patch_size=(9, 9),
        alpha=0.01
    ))


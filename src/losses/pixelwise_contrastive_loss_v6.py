""" This version of the patch-wise contrastive loss directly contrasts the sampled image patches.
    The for-loops are avoided using (re)sampling and the distance matrices.
"""

import torch
import torch.nn.functional as F


def match_loss(patches: torch.Tensor) -> torch.Tensor:
    """ Calculates the match distance from the sample patches.

    :param patches: Tensor in (N, T, K, C, H', W')
    :return:
    """

    N, T, K, _, _, _ = patches.shape

    patches = patches.transpose(1, 2)  # (N, K, T, C, H', W')

    # Get distances across time
    distances = torch.norm(patches.unsqueeze(3) - patches.unsqueeze(2), p=2, dim=[-3, -2, -1])  # (N, K, T, T)

    # Sum distances over time
    distances = torch.sum(distances, dim=[-2, -1])/T  # (N, K)

    # Average across batch and key-points
    return distances.mean()


def non_match_loss(patches: torch.Tensor) -> torch.Tensor:
    """ Calculates the non-match distance from the sampled patches.

    :param patches: Tensor in (N, T, K, C, H', W')
    :return:
    """

    N, T, K, _, _, _ = patches.shape

    # Get distances across key-points
    distances = torch.norm(patches.unsqueeze(3) - patches.unsqueeze(2), p=2, dim=[-3, -2, -1])  # (N, T, K, K)

    # Sum distances over key-points
    distances = torch.sum(distances, dim=[-2, -1])/K  # (N, T)

    # Average across batch and time-steps
    return distances.mean()


def pwcl(
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
        The contrastive loss(es) are calculated by the distance matrices, avoiding the expensive for loops.

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

    """
    # Sample key-point subset
    sample_kpt_ids = torch.randint(low=0,
                                   high=keypoint_coordinates.shape[2],
                                   size=(4, ))

    print(sample_kpt_ids.shape)
    keypoint_coordinates = keypoint_coordinates[:, :, sample_kpt_ids, :]
    print(keypoint_coordinates.shape)
    """

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
    _image_sequence = _image_sequence.view((N*T*K, C, H, W))  # (N*T*K, C, H, W)

    patches = F.grid_sample(
        input=_image_sequence,
        grid=sample_grids,
        mode='bilinear',
        align_corners=False
    ).to(image_sequence.device).view((N, T, K, C, patch_size[0], patch_size[1]))

    match_distance = match_loss(patches)

    non_match_distance = non_match_loss(patches)

    return torch.clamp(match_distance - non_match_distance + alpha, 0)


if __name__ == "__main__":

    N, T, C, H, W = 8, 8, 3, 64, 64

    K = 32

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

    print(pwcl(
        keypoint_coordinates=fake_kpts,
        image_sequence=fake_img,
        pos_range=pos_range,
        grid=grid,
        patch_size=(9, 9),
        alpha=100
    ))


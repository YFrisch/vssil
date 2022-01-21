""" This version of the patch-wise contrastive loss contrasts feature encodings of the sampled image patches.
    The for-loops are avoided using (re)sampling and the distance matrices.
"""
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from kornia.filters import canny


def latent_encodings(patches: torch.Tensor, kpt_coordinates: torch.Tensor) -> torch.Tensor:
    """ Encodes the key-point coordinates and the sampled patches
        into feature vectors.

    :param patches: Tensor in (N, T, K, C, H', W')
    :param kpt_coordinates:  Tensor in (N, T, K, D)
    :return:
    """

    N, T, K, C, H_prime, W_prime = patches.shape
    _, _, _, D = kpt_coordinates.shape

    D_prime = 6

    _patches = (patches + 0.5).clip_(0.0, 1.0)

    encodings = torch.empty((N, T, K, D_prime))

    # First three dimensions of the vector encoding are the latent dimensions (x, y, mu)
    encodings[..., :2] = (kpt_coordinates[..., :2] + 1.0)/2.0
    encodings[..., 2] = kpt_coordinates[..., 2]
    # Average patch intensity over C, H', W'
    encodings[..., 3] = torch.mean(_patches, dim=[-3, -2, -1])
    # Patch gradient information
    magnitudes, edges = canny(
        input=_patches.view((N*T*K, C, H_prime, W_prime)),
        low_threshold=0.2,
        high_threshold=0.5,
        kernel_size=(3, 3)
    )
    magnitudes = magnitudes.view((N, T, K, H_prime, W_prime))
    edges = edges.view((N, T, K, H_prime, W_prime))
    encodings[..., 4] = torch.mean(magnitudes, dim=[-2, -1])
    encodings[..., 5] = torch.mean(edges, dim=[-2, -1])
    # print(torch.mean(encodings, dim=[0, 1, 2]))
    return encodings


def match_loss(encodings: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """ Calculates the match distance from the sample patches.

    :param encodings: Tensor in (N, T, K, C, H', W')
    :return:
    """

    N, T, K, _ = encodings.shape

    encodings = encodings.transpose(1, 2)  # (N, K, T, D')

    # Get distances across time, excluding the latent dimensions
    distances = torch.norm(encodings.unsqueeze(3)[..., 3:] - encodings.unsqueeze(2)[..., 3:], p=2, dim=[-1])  # (N, K, T, T)

    # Sum distances over time
    summed_distances = torch.sum(distances, dim=[-2, -1]) / (T * (T - 1))  # (N, K)

    # Average across batch and key-points
    return summed_distances.mean(), distances


def non_match_loss(encodings: torch.Tensor) -> (torch.Tensor, torch.Tensor):
    """ Calculates the non-match distance from the sampled patches.

    :param encodings: Tensor in (N, T, K, C, H', W')
    :return:
    """

    N, T, K, _ = encodings.shape

    # Get distances across key-points
    distances = torch.norm(encodings.unsqueeze(3) - encodings.unsqueeze(2), p=2, dim=[-1])  # (N, T, K, K)
    # distances = torch.norm(encodings.unsqueeze(3)[..., 3:] - encodings.unsqueeze(2)[..., 3:], p=2, dim=[-1])

    # Sum distances over key-points
    summed_distances = torch.sum(distances, dim=[-2, -1]) / (K * (K - 1))  # (N, T)

    # Average across batch and time-steps
    return summed_distances.mean(), distances


def pwcl(
        keypoint_coordinates: torch.Tensor,
        image_sequence: torch.Tensor,
        grid: torch.Tensor,
        patch_size: tuple = (9, 9),
        alpha: float = 0.1
) -> (torch.Tensor, torch.Tensor, torch.Tensor):

    """ This version of the pixelwise contrastive loss uses torch.grid_sample(...) to extract the (interpolated)
        patches around the key-points.
        The grid is obtained by multiplying a fixed grid, just depending on the input image shape, with the key-points.
        The contrastive loss(es) are calculated by the distance matrices, avoiding the expensive for loops.

    :param keypoint_coordinates: Tensor of key-point coordinates in (N, T, K, 2/3)
    :param image_sequence: Tensor of sequential frames in (N, T, C, H, W)
    :param grid: Tensor of grid positions in (N, H'', W'', 2) where (H'', W'') is the patch size
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

    # Sample surrounding patches
    patches = F.grid_sample(
        input=_image_sequence,
        grid=sample_grids,
        mode='bilinear',
        align_corners=False
    ).to(image_sequence.device).view((N, T, K, C, patch_size[0], patch_size[1]))

    # Get encodings
    encodings = latent_encodings(patches, keypoint_coordinates)

    # Get match and non-match losses
    match_distance, match_distance_matrix = match_loss(encodings)
    non_match_distance, non_match_distance_matrix = non_match_loss(encodings)

    """
    fig, ax = plt.subplots(1, match_distance_matrix.shape[1], figsize=(15, 5))
    for k in range(match_distance_matrix.shape[1]):
        p = ax[k].imshow(match_distance_matrix[0, k, ...])
    fig.colorbar(p)
    plt.suptitle('Match distance matrices')
    plt.show()

    fig, ax = plt.subplots(1, non_match_distance_matrix.shape[1], figsize=(15, 5))
    for t in range(non_match_distance_matrix.shape[1]):
        p = ax[t].imshow(non_match_distance_matrix[0, t, ...])
    fig.colorbar(p)
    plt.suptitle('Non-Match distance matrices')
    plt.show()
    """

    #print('L_match: ', match_distance)
    #print('L_non_match: ', non_match_distance)

    return F.relu(match_distance - non_match_distance + alpha), match_distance, non_match_distance


if __name__ == "__main__":

    device = 'cpu'

    N, T, C, H, W = 8, 8, 3, 64, 64

    K = 32

    patch_size = (9, 9)

    center_index = int(patch_size[0] / 2)

    step_matrix = torch.ones(patch_size + (2,)).to(device)

    step_w = 2 / W
    step_h = 2 / H

    for k in range(0, patch_size[0]):
        for l in range(0, patch_size[1]):
            step_matrix[k, l, 0] = (l - center_index) * step_w
            step_matrix[k, l, 1] = (k - center_index) * step_h

    grid = step_matrix.unsqueeze(0).repeat((N * T * K, 1, 1, 1)).to(device)

    fake_img = torch.rand(size=(N, T, C, H, W)).to(device)

    fake_kpts = torch.rand(size=(N, T, K, 3)).to(device)
    fake_kpts[..., 2] = 1.0

    print(pwcl(
        keypoint_coordinates=fake_kpts,
        image_sequence=fake_img,
        grid=grid,
        patch_size=(9, 9),
        alpha=1.0
    ))


import random

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from kornia.filters import laplacian, canny
from itertools import product


def concat_features_and_keypoints(feature_sequence: torch.Tensor,
                                  key_point_sequence: torch.Tensor) -> torch.Tensor:
    """ Adds the patch endocing to the key-point latent representation.

    :param feature_sequence: Tensor in (N*T*K, D-3)
    :param key_point_sequence: Tensor in (N, T, K, 3)
    :return: Tensor in (N*T*K, D)
    """
    unstacked_shape = key_point_sequence.shape
    key_point_sequence = key_point_sequence.view(feature_sequence.shape)
    feature_sequence = torch.cat([key_point_sequence, feature_sequence], dim=1)
    return feature_sequence.view((*unstacked_shape[:-1], -1))


def get_patch_representation(patch_sequence: torch.Tensor) -> torch.Tensor:
    """ Encodes the extracted image patches into (D-3)-dimensional feature vectors.

    :param patch_sequence: Tensor in (N*T*K, C, H', W')
    :return: Tensor in (N*T*K, D-3)
    """
    n_patches = patch_sequence.shape[0]
    n_features = 3
    features = torch.empty([n_patches, n_features]).to(patch_sequence.device)
    magnitudes, edges = canny(
        input=patch_sequence,
        low_threshold=0.3,
        high_threshold=0.6,
        kernel_size=(3, 3)
    )
    features[:, 0] = torch.mean(patch_sequence, dim=[1, 2, 3])
    features[:, 1] = torch.mean(magnitudes, dim=[1, 2, 3])
    features[:, 2] = torch.mean(edges, dim=[1, 2, 3])
    return features


def match_loss(encodings: torch.Tensor) -> torch.Tensor:
    """ Calculates the match distance from the (sampled) encodings / feature vectors.

        The distance matrices contain the norm differences of each key-point across time.

    :param encodings: Tensor in (N, T, K', D)
    :return:
    """

    # Excluding coordinates for match loss
    _encodings = encodings[..., 3:].permute(0, 2, 1, 3)
    distance_matrix = torch.norm(_encodings.unsqueeze(-2) - _encodings.unsqueeze(-3), dim=-1)
    match_distance = torch.sum(distance_matrix, dim=[-1, -2])

    # Average across batch and key-points
    match_distance = torch.mean(match_distance)

    # TODO: Additionally, the coordinates are smoothed spatially over time, by contrasting their positions
    # positive_distance = torch.norm()

    return match_distance


def non_match_loss(encodings: torch.Tensor) -> torch.Tensor:
    """ Calculates the non-match distance.

    :param encodings: Tensor in (N, T, K', D)
    :return:
    """
    distance_matrix = torch.norm(encodings.unsqueeze(-2) - encodings.unsqueeze(-3), dim=-1)

    # Sum over key-points
    non_match_distance = torch.sum(distance_matrix, dim=[-1, -2])

    # Average over time and batch
    non_match_distance = torch.mean(non_match_distance)

    return non_match_distance


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
    ).to(image_sequence.device)

    feature_encodings = get_patch_representation(patches)  # (N*T*K, D-3)

    feature_encodings = concat_features_and_keypoints(feature_encodings, keypoint_coordinates)  # (N, T, K, D)

    n_samples = 4

    patch_margin = 0.1

    # Sample a sub-set of key-points as anchors
    sample_indices = torch.randint(low=0, high=K, size=(n_samples,))

    sampled_anchor_encodings = feature_encodings[:, :, sample_indices, :]

    match_distance = match_loss(sampled_anchor_encodings)

    non_match_distance = non_match_loss(sampled_anchor_encodings)

    return torch.clamp(match_distance - non_match_distance + patch_margin, 0)


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

    print(pixelwise_contrastive_loss_patch_based(
        keypoint_coordinates=fake_kpts,
        image_sequence=fake_img,
        pos_range=pos_range,
        grid=grid,
        patch_size=(9, 9),
        alpha=0.01
    ))


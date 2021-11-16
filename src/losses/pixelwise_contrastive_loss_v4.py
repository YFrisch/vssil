import random

import torch
import torch.nn.functional as F
from kornia.filters import laplacian, canny
from kornia.morphology import erosion
from itertools import product


def get_feature_representation(feature_map_sequence: torch.Tensor,
                               keypoint_coordinates: torch.Tensor,
                               image_sequence: torch.Tensor,
                               t: int,
                               k: int) -> torch.Tensor:
    N, T, C, H, W = image_sequence.shape

    fts = torch.empty(size=(N, 7)).to(feature_map_sequence.device)
    fts[:, :3] = keypoint_coordinates[:, t, k, ...]
    upscaled_gaussian_map = F.interpolate(feature_map_sequence[:, t: t + 1, k, ...],
                                          size=(H, W))
    mask = (upscaled_gaussian_map > 0.1).float().repeat(1, 3, 1, 1)

    masked_img = torch.multiply(image_sequence[:, t, ...], mask)

    masked_img_magnitude, masked_img_edges = canny(masked_img, low_threshold=0.2,
                                                   high_threshold=0.5, kernel_size=(3, 3))

    # grad_mask = (masked_img_edges > -0.01).float()
    # masked_img_grads = torch.multiply(grad_mask, masked_img_edges)

    fts[:, -1] = masked_img_edges.sum(dim=[1, 2, 3])

    erosion_kernel = torch.ones(3, 3).to(feature_map_sequence.device)

    result_sums = []
    feature_map = torch.clone(mask)
    # while True:
    for _ in range(3):
        result_image = torch.multiply(feature_map, image_sequence[:, t, ...])
        result_sums.append(torch.sum(result_image, dim=[1, 2, 3]))
        feature_map = erosion(feature_map, kernel=erosion_kernel)
        if -1e-2 <= result_sums[-1].mean() <= 1e-2:
            break
    if len(result_sums) >= 3:
        fts[:, 3] = result_sums[-3]
        fts[:, 4] = result_sums[-2]
        fts[:, 5] = result_sums[-1]
    elif len(result_sums) == 2:
        fts[:, 3] = result_sums[-2]
        fts[:, 4] = result_sums[-2]
        fts[:, 5] = result_sums[-1]
    else:
        fts[:, 3] = result_sums[-1]
        fts[:, 4] = result_sums[-1]
        fts[:, 5] = result_sums[-1]

    del mask, masked_img, masked_img_magnitude, masked_img_edges, upscaled_gaussian_map,\
        feature_map, result_image, result_sums, erosion_kernel

    return fts


def pixelwise_contrastive_loss_fmap_based(
        keypoint_coordinates: torch.Tensor,
        image_sequence: torch.Tensor,
        feature_map_sequence: torch.Tensor,
        pos_range: int = 1,
        alpha: float = 0.1
) -> torch.Tensor:
    """ This version of the pixelwise-contrastive loss uses a combination of the feature maps
        with the original images in order to contrast the key-points.

    :param keypoint_coordinates: Tensor of key-point positions in (N, T, K, 2/3)
    :param image_sequence:  Tensor of frames in (N, T, C, H, W)
    :param feature_map_sequence:  Tensor of feature maps in (N, T, K, H', W')
    :param pos_range: Range of time-steps to consider as matches ([anchor - range, anchor + range])
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

    # Calculate contrastive loss
    L = torch.zeros((N,)).to(image_sequence.device)
    for t in range(T):

        for k in range(K):

            #
            #   Anchor feature representation
            #
            anchor_ft_representation = get_feature_representation(
                feature_map_sequence=feature_map_sequence,
                keypoint_coordinates=keypoint_coordinates,
                image_sequence=image_sequence,
                t=t,
                k=k
            )

            time_steps = range(max(0, t - pos_range), min(T, t + pos_range + 1))

            positives = [(t_i, k) for t_i in time_steps]
            positives.remove((t, k))

            negatives = list(product(time_steps, range(K)))
            negatives.remove((t, k))

            for (t_p, k_p) in positives:

                try:
                    negatives.remove((t_p, k_p))
                except ValueError:
                    continue

            #
            #   Match feature representation and loss
            #

            L_p = torch.tensor((N,)).to(image_sequence.device)
            # TODO: Use mining instead of random choice / all positives?
            # for (t_p, k_p) in positives:
            for (t_p, k_p) in [(random.choice(positives) if len(positives) > 1 else positives)]:
                match_ft_representation = get_feature_representation(
                    feature_map_sequence=feature_map_sequence,
                    keypoint_coordinates=keypoint_coordinates,
                    image_sequence=image_sequence,
                    t=t_p,
                    k=k_p
                )

                L_p = L_p + torch.norm(anchor_ft_representation - match_ft_representation, p=2, dim=1) ** 2
            # L_p /= len(positives)

            #
            #   Non-match feature representation and loss
            #

            L_n = torch.tensor((N,)).to(image_sequence.device)
            # TODO: Use mining instead of random choice / all negatives?
            # for (t_n, k_n) in negatives:
            for (t_n, k_n) in [(random.choice(negatives) if len(negatives) > 1 else negatives)]:
                non_match_ft_representation = get_feature_representation(
                    feature_map_sequence=feature_map_sequence,
                    keypoint_coordinates=keypoint_coordinates,
                    image_sequence=image_sequence,
                    t=t,
                    k=k
                )
                L_n = L_n + torch.norm(anchor_ft_representation - non_match_ft_representation, p=2, dim=1) ** 2
            # L_n /= len(negatives)

            #
            #   Total Loss
            #

            L = torch.add(
                L,
                torch.maximum(L_p - L_n + alpha, torch.zeros((N,)).to(image_sequence.device))
            )

    # Average loss across time and key-points
    L = L / (T * K)

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

    print(pixelwise_contrastive_loss_fmap_based(
        keypoint_coordinates=fake_kpts,
        image_sequence=fake_img,
        pos_range=pos_range,
        grid=grid,
        patch_size=(9, 9),
        alpha=0.01
    ))

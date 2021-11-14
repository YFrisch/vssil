import random

import torch
import torch.nn.functional as F
from kornia.filters import laplacian
from itertools import product


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
    :param feature_map_sequence:  Tensor of feature maps in (N, T, C, H', W')
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

            upscaled_anchor_gmap = F.interpolate(feature_map_sequence[:, t: t + 1, k:k + 1, ...],
                                                 size=(H, W))
            anchor_mask = (upscaled_anchor_gmap > 0.1).float()
            masked_anchor_img = torch.multiply(image_sequence[:, t: t + 1, ...], anchor_mask)
            masked_anchor_grads = laplacian(masked_anchor_img, kernel_size=3)

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
            #   Match loss
            #

            L_p = torch.tensor((N,)).to(image_sequence.device)
            # TODO: Use mining instead of random choice / all positives?
            # for (t_p, k_p) in positives:
            for (t_p, k_p) in random.choice(positives):
                match_gaussian_map = F.interpolate(feature_map_sequence[:, t_p: t_p + 1, k_p: k_p + 1, ...],
                                                   size=(H, W))
                match_img_mask = (match_gaussian_map > 0.1).float()
                masked_match_img = torch.multiply(image_sequence[:, t_p: t_p + 1, ...], match_img_mask)
                masked_match_img_grads = laplacian(masked_match_img, kernel_size=3)
                match_grads_mask = (masked_match_img_grads > -0.1).float()
                masked_match_img_grads = masked_match_img_grads * match_grads_mask

                L_p = L_p + torch.norm(torch.mean(masked_anchor_img, dim=[2, 3, 4])
                                       - torch.mean(masked_match_img, dim=[2, 3, 4]), p=2, dim=1)

                L_p = L_p + torch.norm(torch.mean(masked_anchor_grads, dim=[2, 3, 4])
                                       - torch.mean(masked_match_img_grads, dim=[2, 3, 4]), p=2, dim=1)

                L_p = L_p + torch.norm(keypoint_coordinates[:, t, k, ...] - keypoint_coordinates[:, t_p, k_p, ...],
                                       p=2, dim=1) ** 2
            # L_p /= len(positives)

            #
            #   Non-match loss
            #

            L_n = torch.tensor((N,)).to(image_sequence.device)
            # TODO: Use mining instead of random choice / all negatives?
            # for (t_n, k_n) in negatives:
            for (t_n, k_n) in random.choice(negatives):
                non_match_gaussian_map = F.interpolate(feature_map_sequence[:, t_n: t_n + 1, k_n: k_n + 1, ...],
                                                       size=(H, W))
                non_match_img_mask = (non_match_gaussian_map > 0.1).float()
                masked_non_match_img = torch.multiply(image_sequence[:, t_n: t_n + 1, ...], non_match_img_mask)
                masked_non_match_img_grads = laplacian(masked_non_match_img, kernel_size=3)
                non_match_grads_mask = (masked_non_match_img_grads > -0.1).float()
                masked_non_match_img_grads = masked_non_match_img_grads * non_match_grads_mask

                L_n = L_n + torch.norm(torch.mean(masked_anchor_img, dim=[2, 3, 4])
                                       - torch.mean(masked_non_match_img, dim=[2, 3, 4]), p=2, dim=1)

                L_n = L_n + torch.norm(torch.mean(masked_anchor_grads, dim=[2, 3, 4])
                                       - torch.mean(masked_non_match_img_grads, dim=[2, 3, 4]), p=2, dim=1)

                L_n = L_n + torch.norm(keypoint_coordinates[:, t, k, ...] - keypoint_coordinates[:, t_n, k_n, ...],
                                       p=2, dim=1) ** 2
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

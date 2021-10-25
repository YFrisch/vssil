import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from kornia.feature.hardnet import HardNet8
from kornia.feature.tfeat import TFeat
from kornia.morphology import erosion

from src.models.hog_layer import HoGLayer


def get_representation(keypoint_coordinates: torch.Tensor,
                       image: torch.Tensor,
                       feature_map: torch.Tensor) -> torch.Tensor:
    """

    :param keypoint_coordinates: Tensor of key-point coordinates in (N, 2/3)
    :param image: Tensor of current image in (N, C, H, W)
    :param feature_map: Tensor of feature map for key-point in (N, H', W')
    :return:
    """

    N, C, H, W = image.shape

    # Feature maps are converted to 0-1 masks given a threshold
    alpha = 0.5
    mask = torch.round(feature_map).unsqueeze(1)  # (N, H', W'), rounds to the closest integer

    # Use erosion iteratively
    intensities = []
    erosion_kernel = torch.ones(size=(3, 3)).to(image.device)
    _img = mask

    while True:
        _morphed = erosion(_img,
                           kernel=erosion_kernel,
                           engine='convolution')
        _morphed = F.interpolate(input=_morphed, size=(H, W))
        _img = torch.mul(_morphed, image)
        intensity = _img.sum(dim=(1, 2, 3))
        intensities.append(intensity)
        if - 1e-3 <= intensity.mean() <= 1e-3:
            break

    features = torch.empty(size=(image.shape[0], 5)).to(image.device)

    for n in range(image.shape[0]):

        features[n, ...] = torch.tensor([
            keypoint_coordinates[n, 0],
            keypoint_coordinates[n, 1],
            intensities[-1][n],
            intensities[-2][n] if len(intensities) >= 2 else intensities[-1][n],
            intensities[-3][n] if len(intensities) >= 3 else intensities[-1][n]
        ])

    return features


def pixelwise_contrastive_loss(keypoint_coordinates: torch.Tensor,
                               image_sequence: torch.Tensor,
                               feature_map_seq: torch.Tensor,
                               time_window: int = 3,
                               alpha: float = 0.1,
                               verbose: bool = False) -> torch.Tensor:
    """ Encourages key-points to represent different patches of the input image.

    :param keypoint_coordinates: Tensor of key-point coordinates in (N, T, K, 2/3)
    :param image_sequence: Tensor of image sequence in (N, T, C, H, W)
    :param feature_map_seq: Tensor of feature maps per key-point in (N, T, K, H', W')
    :param time_window: Amount of time-steps for positive/negative matching
    :param alpha: Margin for matches vs. non-matches
    :param verbose: Set true for additional output prints
    :return: Tensor of average loss
    """
    assert keypoint_coordinates.dim() == 4
    assert image_sequence.dim() == 5
    assert keypoint_coordinates.shape[0:2] == image_sequence.shape[0:2]
    assert time_window <= image_sequence.shape[1]

    N, T, C, H, W = image_sequence.shape

    K = keypoint_coordinates.shape[2]

    pos_range = max(int(time_window / 2), 1) if time_window > 1 else 0

    # Calculate loss per time-step per key-points
    # The features are extracted and their ids (time-step, # key-point) saved to a list to look them up again if needed
    features = torch.empty(size=(N, T, K, 5)).to(image_sequence.device)
    feature_ids = []

    total_loss = torch.tensor([0.0]).to(image_sequence.device)
    total_loss.requires_grad_(True)

    for t in range(0, T):

        loss_per_timestep = torch.tensor([0.0]).to(image_sequence.device)
        loss_per_timestep.requires_grad_(True)

        for k in range(0, K):

            """
                Anchor patch
                
            """

            matches = [(t_i, k) for t_i in
                       range(max(t - pos_range, 0), min(t + pos_range, T))] if time_window > 1 else []

            non_matches = [(t_j, k_j) for t_j in range(max(t - pos_range, 0), min(t + pos_range, T)) for
                           k_j in range(0, K) if k_j != k]

            # Anchor patch
            if (t, k) in feature_ids:
                anchor_ft = features[:, t, k, ...]
            else:
                anchor_ft = get_representation(keypoint_coordinates=keypoint_coordinates[:, t, k, ...],
                                               image=image_sequence[:, t, ...],
                                               feature_map=feature_map_seq[:, t, k, ...])
                features[:, t, k, ...] = anchor_ft
                feature_ids.append((t, k))

            """
                Match (positive) patches
            
            """

            L_match = torch.tensor([0.0]).to(image_sequence.device)
            L_match.requires_grad_(True)

            for t_i, k_i in matches:
                if (t_i, k_i) in feature_ids:
                    match_ft = features[:, t_i, k_i, ...]
                else:
                    match_ft = get_representation(keypoint_coordinates=keypoint_coordinates[:, t_i, k_i, ...],
                                                  image=image_sequence[:, t_i, ...],
                                                  feature_map=feature_map_seq[:, t_i, k_i, ...])
                    features[:, t_i, k_i, ...] = match_ft
                    feature_ids.append((t_i, k_i))

                L_match = L_match + torch.norm(anchor_ft - match_ft, p=2)
            L_match = L_match / len(matches)

            """
                Non-match (negative) patches
            
            """

            L_non_match = torch.tensor([0.0]).to(image_sequence.device)
            L_non_match.requires_grad_(True)

            for t_j, k_j in non_matches:
                if (t_j, k_j) in feature_ids:
                    non_match_ft = features[:, t_j, k_j, ...]
                else:
                    non_match_ft = get_representation(
                        keypoint_coordinates=keypoint_coordinates[:, t_j, k_j, ...],
                        image=image_sequence[:, t_j, ...],
                        feature_map=feature_map_seq[:, t_j, k_j, ...])
                    features[:, t_j, k_j, ...] = non_match_ft
                    feature_ids.append((t_j, k_j))

                L_non_match = L_non_match + torch.norm(anchor_ft - non_match_ft, p=2)
            L_non_match = L_non_match / len(non_matches)

            loss_per_timestep = loss_per_timestep + \
                                (max(L_match - L_non_match + alpha, torch.tensor([0.0]).to(image_sequence.device)))
            loss_per_timestep = loss_per_timestep / (K * (time_window * K - 1))

        total_loss = total_loss + loss_per_timestep

    return total_loss

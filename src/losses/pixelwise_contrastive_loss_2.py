import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from kornia.feature.hardnet import HardNet8
from kornia.feature.tfeat import TFeat

from src.models.hog_layer import HoGLayer


def get_patch_by_gridsampling(keypoint_coordinates: torch.Tensor,
                              image: torch.Tensor,
                              patch_size: tuple):
    N, C, H, W = image.shape

    grid = torch.zeros(size=(N, patch_size[0], patch_size[1], 2)).to(image.device)

    center = int(grid.shape[1] / 2)
    grid[:, center, center, 0] = - keypoint_coordinates[:, 1]  # Height
    grid[:, center, center, 1] = keypoint_coordinates[:, 0]  # Width
    step_h = (1 / H)
    step_w = (1 / W)

    for h_i in range(grid.shape[1]):
        for w_i in range(grid.shape[2]):
            step_size_h = h_i - center
            step_size_w = w_i - center
            grid[:, h_i, w_i, 0] = keypoint_coordinates[:, 0] + step_size_h * step_h
            grid[:, h_i, w_i, 1] = keypoint_coordinates[:, 1] + step_size_w * step_w

    # patches = F.grid_sample(input=image, grid=grid, padding_mode='zeros', align_corners=False)
    # patches = F.grid_sample(input=image, grid=grid, padding_mode='border', align_corners=False)
    patches = F.grid_sample(input=image, grid=grid, padding_mode='reflection', align_corners=False)

    assert tuple(patches.shape[-2:]) == patch_size, f'{tuple(patches.shape[-2:])} != {patch_size}'
    assert patches.dim() == 4

    return patches


'''
def get_image_patch(keypoint_coordinates: torch.Tensor,
                    image: torch.Tensor,
                    patch_size: tuple):
    """ Return the patch of size (H', W') of the input image around the input key-point.

    :param keypoint_coordinates: Coordinates (batch) of a single key-point in (N, 2/3)
    :param image: Image (batch) to extract patch from in (N, C, H, W)
    :param patch_size: Shape of the image patch
    :return: Extracted image patch in (N, C, H', W')
    """

    assert keypoint_coordinates.dim() == 2
    assert keypoint_coordinates.shape[1] in [2, 3]
    assert image.dim() == 4
    assert image.shape[0] == keypoint_coordinates.shape[0]
    assert patch_size[0] <= image.shape[2]
    assert patch_size[1] <= image.shape[3]

    N, _, H, W = image.shape

    # Convert key-point representation to image pixel coordinates
    h = (keypoint_coordinates[:, 0] + 1) / 2 * H  # (N, 1)
    w = (-keypoint_coordinates[:, 1] + 1) / 2 * W  # (N, 1

    # Calculate ranges of image patch while ensuring image boundaries
    zeros = torch.zeros_like(h)
    max_height = torch.ones_like(h) * H
    max_width = torch.ones_like(w) * W
    h_min = torch.floor(torch.maximum(h - int(patch_size[0] / 2), zeros)) + 1
    h_max = torch.floor(torch.minimum(h + int(patch_size[0] / 2), max_height)) + 1
    w_min = torch.floor(torch.maximum(h - int(patch_size[1] / 2), zeros)) + 1
    w_max = torch.floor(torch.minimum(h + int(patch_size[1] / 2), max_width)) + 1

    # Extract patch (individually per entry of batch)
    patches = None
    for n in range(N):
        patch = image[n:n + 1, :, h_min[n].long():h_max[n].long(), w_min[n].long():w_max[n].long()]
        # If patch does not have query size, interpolate to query size
        if not tuple(patch.shape[-2:]) == patch_size:
            patch = F.interpolate(patch, size=patch_size, align_corners=False, mode='bilinear')
        patches = patch if patches is None else torch.cat([patches, patch], dim=0)

    assert tuple(patches.shape[-2:]) == patch_size, f'{tuple(patches.shape[-2:])} != {patch_size}'
    assert patches.dim() == 4

    return patches
'''


def patch_diff(anchor_patch: torch.Tensor,
               contrast_patch: torch.Tensor,
               anchor_keypoint_coords: torch.Tensor = None,
               contrast_keypoint_coords: torch.Tensor = None,
               mode: str = 'norm'):
    """ Returns the distance between anchor and the patch to contrast.

    :param anchor_patch: The anchor image patch in (N, C, H', W')
    :param contrast_patch: The patch to compare to in (N, C, H', W')
    :param anchor_keypoint_coords: Tensor of current anchor key-point's coordinates in (N, 2/3)
    :param contrast_keypoint_coords: Tensor of key-point coordinates that are to contrast to anchor in (N, 2/3)
    :param mode: What difference/distance measure to use
    :return: Patch difference
    """
    if mode == 'norm':
        return torch.norm(input=(anchor_patch - contrast_patch), p=2)
    elif mode == 'vssil':
        assert anchor_keypoint_coords is not None and contrast_keypoint_coords is not None, \
            "Key point coordinates are required for this mode."

        N, C, Hp, Wp = anchor_patch.shape
        center_height = int(Hp / 4)
        center_width = int(Wp / 4)
        center_h = int(Hp / 2)
        center_w = int(Wp / 2)
        center_mask = torch.zeros_like(anchor_patch)
        center_mask[:, :, center_h - center_height: center_h + center_height,
        center_w - center_width: center_w + center_width] = 1
        off_center_mask = torch.ones_like(anchor_patch) - center_mask

        batch_anchor_features = None
        batch_contrast_features = None
        for n in range(anchor_keypoint_coords.shape[0]):
            anchor_features = torch.tensor([
                anchor_keypoint_coords[n, 0],
                anchor_keypoint_coords[n, 1],
                anchor_keypoint_coords[n, 2] if anchor_keypoint_coords.shape[1] == 3 else 0,
                (1 / Hp) * torch.sum(anchor_patch[n, :] * center_mask),
                (1 / Hp ** 2) * torch.sum(anchor_patch[n, :] * off_center_mask)
            ])
            contrast_features = torch.tensor([
                contrast_keypoint_coords[n, 0],
                contrast_keypoint_coords[n, 1],
                contrast_keypoint_coords[n, 2] if contrast_keypoint_coords.shape[1] == 3 else 0,
                (1 / Hp) * torch.sum(contrast_patch[n, :] * center_mask),
                (1 / Hp ** 2) * torch.sum(contrast_patch[n, :] * off_center_mask)
            ])
            batch_anchor_features = anchor_features.unsqueeze(0) if batch_anchor_features is None \
                else torch.cat([batch_anchor_features, anchor_features.unsqueeze(0)])
            batch_contrast_features = contrast_features.unsqueeze(0) if batch_contrast_features is None \
                else torch.cat([batch_contrast_features, contrast_features.unsqueeze(0)])
            assert batch_anchor_features.shape == batch_contrast_features.shape
            return torch.norm(input=batch_anchor_features - batch_contrast_features, p=2)
    elif mode == 'hog':
        stretched_img_shape = (anchor_patch.shape[2] * 2, anchor_patch.shape[3])
        hog_layer = HoGLayer(img_shape=stretched_img_shape,
                             cell_size=(2, 1)).to(anchor_patch.device)
        stretched_anchor_patch = F.interpolate(anchor_patch, stretched_img_shape)
        strechted_contrast_patch = F.interpolate(contrast_patch, stretched_img_shape)
        return torch.norm(input=hog_layer(stretched_anchor_patch)[0] - hog_layer(strechted_contrast_patch)[0], p=2)
    elif mode == 'HardNet8':
        hard_net8 = HardNet8()
        grey = T.Grayscale()
        upscaled_anchor_patch = F.interpolate(anchor_patch, size=(32, 32))
        grey_upscaled_anchor_patch = grey(upscaled_anchor_patch)
        upscaled_contrast_patch = F.interpolate(contrast_patch, size=(32, 32))
        gray_upscaled_contrast_patch = grey(upscaled_contrast_patch)
        with torch.no_grad():
            anchor_ft = hard_net8(grey_upscaled_anchor_patch)
            contrast_ft = hard_net8(gray_upscaled_contrast_patch)
        return torch.norm(input=anchor_ft - contrast_ft, p=2)
    elif mode == 'TFeat':
        tfeat = TFeat()
        grey = T.Grayscale()
        upscaled_anchor_patch = F.interpolate(anchor_patch, size=(32, 32))
        grey_upscaled_anchor_patch = grey(upscaled_anchor_patch)
        upscaled_contrast_patch = F.interpolate(contrast_patch, size=(32, 32))
        gray_upscaled_contrast_patch = grey(upscaled_contrast_patch)
        anchor_ft = tfeat(grey_upscaled_anchor_patch)
        contrast_ft = tfeat(gray_upscaled_contrast_patch)
        return torch.norm(input=anchor_ft - contrast_ft, p=2)
    else:
        raise NotImplemented("Unknown patch distance method.")


def pixelwise_contrastive_loss(keypoint_coordinates: torch.Tensor,
                               image_sequence: torch.Tensor,
                               patch_size: tuple = (8, 8),
                               time_window: int = 3,
                               alpha: float = 0.1,
                               patch_diff_mode: str = 'norm',
                               verbose: bool = False) -> torch.Tensor:
    """ Encourages key-points to represent different patches of the input image.

    :param keypoint_coordinates: Tensor of key-point coordinates in (N, T, K, 2/3)
    :param image_sequence: Tensor of image sequence in (N, T, C, H, W)
    :param patch_size: Size of the image patch around each key-point position
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

    patches = torch.empty(size=(N, T, K, C, patch_size[0], patch_size[1]))
    patches_ids = []
    # patches = {}

    # Calculate loss per time-step per key-points
    # The patches are extracted online and dynamically
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
            if (t, k) in patches_ids:
                anchor_patch = patches[:, t, k, ...]
            else:
                anchor_patch = get_patch_by_gridsampling(keypoint_coordinates=keypoint_coordinates[:, t, k, ...],
                                                         image=image_sequence[:, t, ...],
                                                         patch_size=patch_size)
                patches[:, t, k, ...] = anchor_patch

            """
                Match (positive) patches
            
            """

            L_match = torch.tensor([0.0]).to(image_sequence.device)
            L_match.requires_grad_(True)
            for t_i, k_i in matches:
                if (t_i, k_i) in patches_ids:
                    match_patch = patches[:, t_i, k_i, ...]
                else:
                    match_patch = get_patch_by_gridsampling(keypoint_coordinates=keypoint_coordinates[:, t_i, k_i, ...],
                                                            image=image_sequence[:, t_i, ...],
                                                            patch_size=patch_size)
                    patches[:, t_i, k_i, ...] = match_patch

                L_match = L_match + patch_diff(anchor_patch=anchor_patch,
                                               contrast_patch=match_patch,
                                               anchor_keypoint_coords=keypoint_coordinates[:, t, k, ...],
                                               contrast_keypoint_coords=keypoint_coordinates[:, t_i, k_i, ...],
                                               mode=patch_diff_mode)
            L_match = L_match / len(matches)

            """
                Non-match (negative) patches
            
            """

            L_non_match = torch.tensor([0.0]).to(image_sequence.device)
            L_non_match.requires_grad_(True)
            for t_j, k_j in non_matches:
                if (t_j, k_j) in patches_ids:
                    non_match_patch = patches[:, t_j, k_j, ...]
                else:
                    non_match_patch = get_patch_by_gridsampling(
                        keypoint_coordinates=keypoint_coordinates[:, t_j, k_j, ...],
                        image=image_sequence[:, t_j, ...],
                        patch_size=patch_size)
                    patches[:, t_j, k_j, ...] = non_match_patch
                L_non_match = L_non_match + patch_diff(anchor_patch=anchor_patch,
                                                       contrast_patch=non_match_patch,
                                                       anchor_keypoint_coords=keypoint_coordinates[:, t, k, ...],
                                                       contrast_keypoint_coords=keypoint_coordinates[:, t_j, k_j, ...],
                                                       mode=patch_diff_mode)
            L_non_match = L_non_match / len(non_matches)

            loss_per_timestep = loss_per_timestep + \
                                (max(L_match - L_non_match + alpha, torch.tensor([0.0]).to(image_sequence.device)))
            loss_per_timestep = loss_per_timestep / (K * (time_window * K - 1))

        total_loss = total_loss + loss_per_timestep

    return total_loss

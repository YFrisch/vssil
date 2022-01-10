import os

import matplotlib.pyplot as plt
import torch
from torchvision.io import read_video

from src.losses.pixelwise_contrastive_loss_v7 import pwcl
from src.losses.kpt_metrics import tracking_metric, distribution_metric, visual_difference_metric
from test_keypoints import get_perfect_keypoints, \
    get_bad_keypoints, get_random_keypoints


def kpts_2_img_coordinates(kpt_coordinates: torch.Tensor,
                           img_shape: tuple) -> torch.Tensor:
    """ Converts the key-point coordinates from video structure format [-1,1]x[-1,1]
        to image coordinates in [0,W]x[0,H].
    """
    _kpt_coordinates = kpt_coordinates.clone()
    _kpt_coordinates[..., 1] = (kpt_coordinates[..., 1] + 1.0) * img_shape[-2]/2.0
    _kpt_coordinates[..., 0] = (-kpt_coordinates[..., 0] + 1.0) * img_shape[-1]/2.0
    return _kpt_coordinates


def load_sample_images(sample_size: int = 4,
                       time_steps: int = 2,
                       path: str = 'tests/contrastive_loss_test_data/990000.mp4') -> torch.Tensor:

    vid = read_video(path, pts_unit='sec')[0]
    rand_start_ind = torch.randint(low=0, high=vid.shape[0]-sample_size*time_steps, size=(1, ))
    vid = vid[rand_start_ind:rand_start_ind+sample_size*time_steps:time_steps, ...]

    return vid.permute(0, 3, 1, 2)/255.0


if __name__ == "__main__":

    torch.manual_seed(123)

    sample_size = 4
    batch_size = 16

    img_tensor = load_sample_images(sample_size=sample_size).unsqueeze(0)
    img_tensor = img_tensor.repeat((batch_size, 1, 1, 1, 1))
    N, T, C, H, W = img_tensor.shape

    time_window = 5
    patch_size = (35, 35)

    pos_range = max(int(time_window / 2), 1) if time_window > 1 else 0
    center_index = int(patch_size[0] / 2)

    step_matrix = torch.ones(patch_size + (2,))

    step_w = 2 / W
    step_h = 2 / H

    for k in range(0, patch_size[0]):
        for l in range(0, patch_size[1]):
            step_matrix[k, l, 0] = (l - center_index) * step_w
            step_matrix[k, l, 1] = (k - center_index) * step_h

    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].imshow(img_tensor[0, 0].permute(1, 2, 0))
    ax[0].axis('off')
    ax[1].imshow((img_tensor[0, 0] - img_tensor[0, -1] + 0.5).clip(0.0, 1.0).permute(1, 2, 0))
    ax[1].axis('off')
    ax[2].imshow(img_tensor[0, -1].permute(1, 2, 0))
    ax[2].axis('off')
    plt.show()

    kpt_coordinates = get_perfect_keypoints(T=sample_size).unsqueeze(0)
    kpt_coordinates = kpt_coordinates.repeat((batch_size, 1, 1, 1))
    _, _, K, D = kpt_coordinates.shape

    kpts = kpts_2_img_coordinates(kpt_coordinates, tuple(img_tensor.shape[-2:]))
    fig, ax = plt.subplots(1, sample_size, figsize=(15, 5))
    for t in range(sample_size):
        ax[t].imshow(img_tensor[0, t].permute(1, 2, 0))
        ax[t].scatter(kpts[0, t, :, 1], kpts[0, t, :, 0], color='red')
    plt.show()
    exit()

    print("\n##### Perfect Key-Points:")
    grid = step_matrix.unsqueeze(0).repeat((N * T * K, 1, 1, 1))

    L_pwc = pwcl(
        keypoint_coordinates=kpt_coordinates,
        image_sequence=img_tensor,
        grid=grid,
        pos_range=pos_range,
        patch_size=patch_size,
        alpha=0.4
    )
    print(L_pwc)

    print("\n##### Bad Key-Points:")
    kpt_coordinates = get_bad_keypoints(T=sample_size).unsqueeze(0)
    kpt_coordinates = kpt_coordinates.repeat((batch_size, 1, 1, 1))
    L_pwc = pwcl(
        keypoint_coordinates=kpt_coordinates,
        image_sequence=img_tensor,
        grid=grid,
        pos_range=pos_range,
        patch_size=patch_size,
        alpha=0.4
    )
    print(L_pwc)

    print("\n##### Random Key-Points:")
    kpt_coordinates = get_random_keypoints(T=sample_size).unsqueeze(0)
    kpt_coordinates = kpt_coordinates.repeat((batch_size, 1, 1, 1))
    L_pwc = pwcl(
        keypoint_coordinates=kpt_coordinates,
        image_sequence=img_tensor,
        grid=grid,
        pos_range=pos_range,
        patch_size=patch_size,
        alpha=0.4
    )
    print(L_pwc)

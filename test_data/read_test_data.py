"""
    This script contains functionalities to load the image sequences
    and key-point coordinates from the test data at data/

"""

import torch
import matplotlib.pyplot as plt
from torchvision.io import read_video

from src.utils.visualization import imprint_img_with_kpts


def read_image_sequence(mp4_path: str, sample_freq: int = 1) -> torch.Tensor:
    img_tensor, _, _ = read_video(filename=mp4_path, pts_unit='sec')
    img_tensor = img_tensor[::sample_freq, ...].permute(0, 3, 1, 2)
    return img_tensor  # (T, C, H, W)


def read_key_point_coordinates(kpt_paths: [str], sample_freq: int = 1) -> torch.Tensor:
    all_kpts = None
    for kpt_path in kpt_paths:
        with open(kpt_path) as kpt_file:
            lines = kpt_file.readlines()
            kpt_tensor = torch.empty(size=(len(lines), 1, 3))
            for line_nr, line in enumerate(lines):
                x, y, i = line.split(";")
                x, y, i = float(x), float(y), float(i)
                kpt_tensor[line_nr, 0, :] = torch.tensor([x, y, i])
        all_kpts = kpt_tensor if all_kpts is None else torch.cat([all_kpts, kpt_tensor], dim=1)
    all_kpts = all_kpts[::sample_freq, ...]
    return all_kpts


def convert_img_coordinates_to_kpts(kpt_coordinates: torch.Tensor, img_shape: tuple) -> torch.Tensor:
    kpts = kpt_coordinates.clone()
    print(kpts[100, ...])
    kpts[..., 1] = ((kpts[..., 0] / img_shape[-1]) * 2) - 1  # W
    kpts[..., 0] = ((kpts[..., 1] / img_shape[-2]) * 2) - 1  # H
    print(kpts[100, ...])
    return kpts


if __name__ == "__main__":
    img_series_tensor = read_image_sequence(
        mp4_path="/home/yannik/vssil/test_data/smooth_noisy_1.mp4",
        sample_freq=5
    )
    print(f"Loaded image series tensor of shape {img_series_tensor.shape}.")

    kpt_coordinates_tensor = read_key_point_coordinates(
        ['/home/yannik/vssil/test_data/smooth_noisy_1_0.txt',
         '/home/yannik/vssil/test_data/smooth_noisy_1_1.txt',
         '/home/yannik/vssil/test_data/smooth_noisy_1_2.txt'],
        sample_freq=5
    )

    print(f"Loaded key-point coordinates tensor of shape {kpt_coordinates_tensor.shape}.")

    kpt_tensor = convert_img_coordinates_to_kpts(kpt_coordinates_tensor,
                                                 tuple(img_series_tensor.shape[-2:]))

    time_step = 100

    plt.imshow(img_series_tensor[time_step, ...].permute(1, 2, 0).cpu().numpy())

    imprinted_frame = imprint_img_with_kpts(
        img=img_series_tensor[time_step: time_step + 1, ...].unsqueeze(0),
        kpt=kpt_tensor[time_step: time_step + 1, ...].unsqueeze(0)
    )

    print(f"Plotting frame at t={time_step} with key-points:")

    plt.imshow(imprinted_frame)
    plt.show()

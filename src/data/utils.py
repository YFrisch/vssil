import os.path

import torch
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from torchvision import transforms
from torch.utils.data import Dataset

from .npz_dataset import NPZ_Dataset
from .video_dataset import VideoFrameDataset, ImglistToTensor


def play_video(video_series: torch.Tensor):
    """ Plays given torch tensor in
        (time, channel, height, width) format
         as video / matplotlib animation.

    :param video_series: The torch tensor in (T, C, H, W) format.
    """

    assert video_series.dim() == 4, "Input video does not have 4 dimensions."

    video_series = video_series.permute(0, 2, 3, 1).detach().cpu().numpy()  # (T, H, W, C)

    c = video_series.shape[-1]

    fig = plt.figure()
    frame = np.zeros((video_series.shape[1], video_series.shape[2], video_series.shape[3]))
    if c == 1:
        im = plt.imshow(frame.squeeze(), cmap='Greys', vmin=0, vmax=1)
    else:
        im = plt.imshow(frame)

    def init():
        if c == 1:
            im.set_data(video_series[0, ...].squeeze())
        else:
            im.set_data(video_series[0, ...])

    def animate(i: int):
        frame = video_series[i, ...].squeeze()
        im.set_data(frame)
        return im

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=video_series.shape[0], interval=100)

    plt.show()


def get_dataset_from_path(root_path: str, n_timesteps: int) -> Dataset:
    """ Creates and returns the appropriate type of data-set, depending on the root path.

    :param root_path: Path to the dataset
    :param n_timesteps: Sequential length of samples
    """

    if root_path.endswith(".npz"):
        data_set = NPZ_Dataset(
            num_timesteps=n_timesteps,
            root_path=root_path,
            key_word='images'
        )
    elif os.path.isdir(root_path):
        preprocess = transforms.Compose([
            # NOTE: This first transform already converts the image range to (0, 1)
            ImglistToTensor(),

        ])
        data_set = VideoFrameDataset(
            root_path=root_path,
            annotationfile_path=os.path.join(root_path, 'annotations.txt'),
            num_segments=1,
            frames_per_segment=n_timesteps,
            imagefile_template='img_{:05d}.jpg',
            transform=preprocess,
            random_shift=False,
            test_mode=True
        )
    else:
        raise ValueError(f"Invalid root path at {root_path}")

    return data_set

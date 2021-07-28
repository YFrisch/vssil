import torch
from torch.utils.data import ConcatDataset
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation

from old.src.data.mime import MimeHDKinectRGB


def play_video(video_series: torch.Tensor):
    """ Plays given torch tensor in
        (time, channel, height, width) format
         as video / matplotlib animation.

    :param video_series: The torch tensor in (T, C, H, W) format.
    """

    assert video_series.dim() == 4, "Input video does not have 4 dimensions."

    video_series = video_series.permute(0, 2, 3, 1).detach().cpu().numpy()  # (T, H, W, C)
    # video_series = video_series.permute(0, 2, 3, 1).cpu().numpy()

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


def combine_mime_hd_kinect_tasks(task_list: [str], base_path: str,
                                 start_ind: int = 0, stop_ind: int = -1,
                                 timesteps_per_sample: int = -1, overlap: int = 20,
                                 img_shape: (float, float) = (1.0, 1.0)):
    list_of_datasets = []
    for task in task_list:
        dataset = MimeHDKinectRGB(
            base_path, task, start_ind, stop_ind,
            timesteps_per_sample, overlap, img_shape
        )
        list_of_datasets.append(dataset)

    return ConcatDataset(list_of_datasets)

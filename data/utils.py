from time import sleep

import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation


def play_video(video_series: torch.Tensor):
    """ Plays given torch tensor in
        (time, channel, height, width) format
         as video / matplotlib animation.

    :param video_series: The torch tensor in (T, C, H, W) format.
    """

    assert video_series.dim() == 4, "Input video does not have 4 dimensions."

    video_series = video_series.permute(0, 2, 3, 1).detach().cpu().numpy()
    # video_series = video_series.permute(0, 2, 3, 1).cpu().numpy()

    # TODO: Not working yet for greyscale image series.
    c = video_series.shape[-1]

    fig = plt.figure()
    frame = np.zeros((video_series.shape[1], video_series.shape[2], video_series.shape[3]))
    if c == 1:
        im = plt.imshow(frame.squeeze(), cmap='Greys')
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


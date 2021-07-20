import torch
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.collections import PathCollection
import numpy as np


def make_annotated_tensor(image_series: torch.Tensor, feature_positions: torch.Tensor):
    """ Modifies the input tensor cointaining a sequence of images to
        be annotated at the given feature positions
    :param image_series: Torch tensor of image series in (T, C, H, W) format
    :param feature_positions: Torch tensor of 2D feature positions in (T, F, 2) format
    """

    new_img_series = torch.clone(image_series)
    size = 2
    for t in range(image_series.shape[0]):
        for f in range(feature_positions.shape[1]):
            feature_pos_2d = feature_positions[t, f, :].int()
            x_start = feature_pos_2d[0] - size
            x_stop = feature_pos_2d[0] + size
            y_start = feature_pos_2d[1] - size
            y_stop = feature_pos_2d[1] + size
            new_img_series[t, 0, x_start:x_stop, y_start:y_stop] = 0  # R
            new_img_series[t, 1, x_start:x_stop, y_start:y_stop] = 1  # G
            new_img_series[t, 2, x_start:x_stop, y_start:y_stop] = 0  # B

    return new_img_series


def play_series_with_keypoints(image_series: torch.Tensor, keypoint_coords: torch.Tensor):
    """ Plots the given image series together with the keypoint coordinates.

    :param image_series: Image series tensor in (N, T, C, H, W)
    :param keypoint_coords: Torch tensor of series of keypoint coordinates in (N, T, C, 3)
                            where C is the number of key-points and the last dimension is its
                            (x-coordinate (%), y-coordinate (%), intensity)
    :return: None
    """
    assert image_series.dim() == 5, "Wrong shape of input image series!"
    assert keypoint_coords.dim() == 4, "Wrong shape of input key-point coordinate series!"
    assert image_series.shape[0:2] == keypoint_coords.shape[0:2], "Batch size or time-steps do not match!"

    N, T = image_series.shape[0], image_series.shape[1]
    C = image_series.shape[2]
    assert C in [1, 3], "Only one or three channels supported for image series!"

    image_width = image_series.shape[3]
    image_height = image_series.shape[4]

    # Permute to (N, T, H, W, C) for matplotlib
    image_series = image_series.permute(0, 1, 3, 4, 2).detach().cpu().numpy()

    frame = np.zeros((image_width, image_height, C))

    n2 = np.min([N, 4])
    n1 = int(np.ceil(N/4))
    fig, ax = plt.subplots(n1, n2)

    im_objects = []

    for n in range(N):
        n_k = n % 4
        n_l = int(np.floor(n / 4))
        if C == 1:
            im = ax[n_l, n_k].imshow(frame.squeeze(), cmap='Greys', vmin=0, vmax=1)
        else:
            im = ax[n_l, n_k].imshow(frame)
        im_objects.append(im)

    def init():
        for n_i in range(N):
            if C == 1:
                im_objects[n_i].set_data(image_series[n_i, 0, ...].squeeze())
            else:
                im_objects[n_i].set_data(image_series[n_i, 0, ...])

    def animate(t: int):
        for n_i in range(N):
            frame = image_series[n_i, t, ...].squeeze()
            im_objects[n_i].set_data(frame)
            return im_objects

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=T, interval=100)

    plt.show()


def play_series_and_reconstruction_with_keypoints(image_series: torch.Tensor,
                                                  reconstruction: torch.Tensor,
                                                  keypoint_coords: torch.Tensor):

    assert image_series.dim() == reconstruction.dim() == 5
    assert keypoint_coords.dim() == 4
    assert image_series.shape[0:2] == keypoint_coords.shape[0:2] == reconstruction.shape[0:2],\
        "Batch sizes or time-steps do not match!"

    N, T = image_series.shape[0], image_series.shape[1]
    C = image_series.shape[2]
    assert C in [1, 3], "Only one or three channels supported for image series!"
    assert N == 1, "Only one sample supported (Batch size of 1)."

    image_width = image_series.shape[3]
    image_height = image_series.shape[4]

    # Permute to (N, T, H, W, C) for matplotlib
    image_series = image_series.permute(0, 1, 3, 4, 2).detach().cpu().numpy()
    reconstruction = reconstruction.permute(0, 1, 3, 4, 2).detach().cpu().numpy()

    frame = np.zeros((image_width, image_height, C))

    fig, ax = plt.subplots(1, 2)
    if C == 1:
        orig_im_obj = ax[0].imshow(frame.squeeze(), cmap='Greys', vmin=0, vmax=1)
        rec_im_obj = ax[1].imshow(frame.squeeze(), cmap='Greys', vmin=0, vmax=1)
    else:
        orig_im_obj = ax[0].imshow(frame)
        rec_im_obj = ax[1].imshow(frame)

    # orig_scatter_obj = PathCollection()
    # rec_scatter_obj = PathCollection()

    """
    def init():
        if C == 1:
            orig_im_obj.set_data(image_series[0, 0, ...].squeeze())
            rec_im_obj.set_data(reconstruction[0, 0, ...].squeeze())
        else:
            orig_im_obj.set_data(image_series[0, 0, ...])
            rec_im_obj.set_data(reconstruction[0, 0, ...])
        for n_keypoints in range(keypoint_coords.shape[2]):
            x = int((keypoint_coords[0, 0, n_keypoints, 0] + 1)/2 * image_height)
            y = int((keypoint_coords[0, 0, n_keypoints, 1] + 1)/2 * image_width)
            intensity = keypoint_coords[0, 0, n_keypoints, 2]
            if intensity > 0.5:
                orig_scatter_obj = ax[0].scatter(x, y,)
                rec_scatter_obj = ax[1].scatter(x, y)
    """
    orig_scatter_objects = []
    rec_scatter_objects = []
    if C == 1:
        orig_im_obj.set_data(image_series[0, 0, ...].squeeze())
        rec_im_obj.set_data(reconstruction[0, 0, ...].squeeze())
    else:
        orig_im_obj.set_data(image_series[0, 0, ...])
        rec_im_obj.set_data(reconstruction[0, 0, ...])
    for n_keypoint in range(keypoint_coords.shape[2]):
        x = int((keypoint_coords[0, 0, n_keypoint, 0] + 1) / 2 * image_height)
        y = int((keypoint_coords[0, 0, n_keypoint, 1] + 1) / 2 * image_width)
        intensity = keypoint_coords[0, 0, n_keypoint, 2]
        if intensity > 0.0:
            orig_scatter_obj = ax[0].scatter(0, 0)
            rec_scatter_obj = ax[1].scatter(0, 0)
            orig_scatter_objects.append(orig_scatter_obj)
            rec_scatter_objects.append(rec_scatter_obj)

    def animate(t: int):
        im_frame = image_series[0, t, ...].squeeze()
        orig_im_obj.set_data(im_frame)
        rec_frame = reconstruction[0, t, ...].squeeze()
        rec_im_obj.set_data(rec_frame)
        for n_keypoint in range(keypoint_coords.shape[2]):
            x = int((keypoint_coords[0, t, n_keypoint, 0] + 1)/2 * image_height)
            y = int((keypoint_coords[0, t, n_keypoint, 1] + 1)/2 * image_width)
            intensity = keypoint_coords[0, t, n_keypoint, 2]
            if intensity > 0.1:
                orig_scatter_objects[n_keypoint].set_offsets([x, y])
                rec_scatter_objects[n_keypoint].set_offsets([x, y])
        return im_frame, rec_frame, orig_scatter_objects, rec_scatter_objects

    anim = animation.FuncAnimation(fig, animate, frames=T, interval=100, repeat=False)

    plt.show()




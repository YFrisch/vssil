import io
import os.path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, animation
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image
from torchvision import transforms
from cv2 import VideoWriter, VideoWriter_fourcc, \
    normalize, NORM_MINMAX, CV_32F
import pylab

from src.losses.kpt_metrics import patchwise_contrastive_metric
from src.utils.kpt_utils import kpts_2_img_coordinates


def numpy_to_mp4(img_array: np.ndarray, target_path: str = 'test.avi', fps: int = 20):
    """ Takes a numpy array in (T, H, W, C) format and makes a video out of it."""
    width = img_array.shape[2]
    height = img_array.shape[1]

    # Convert to 255 RGB
    norm_img_array = normalize(img_array, None, alpha=0, beta=255,
                               norm_type=NORM_MINMAX, dtype=CV_32F)
    norm_img_array = norm_img_array.astype(np.uint8)
    assert img_array.shape[0] % fps == 0
    sec = int(img_array.shape[0] / fps)
    fourcc = VideoWriter_fourcc(*'MPEG')
    video = VideoWriter(target_path, fourcc, float(fps), (width, height))
    for frame_count in range(fps * sec):
        video.write(norm_img_array[frame_count, ...])
    video.release()


def play_series_with_keypoints(image_series: torch.Tensor,
                               keypoint_coords: torch.Tensor,
                               intensity_threshold: float = 0.9,
                               key_point_trajectory: bool = False,
                               trajectory_length: int = 10,
                               save_path: str = ".",
                               save_frames: bool = False):
    """ Visualizes the image-series tensor together with the given predicted key-points. """
    assert keypoint_coords.dim() == 4
    (N, T, C, H, W) = tuple(image_series.shape)
    assert C in [1, 3], "Only one or three channels supported for plotting image series!"
    assert N == 1, "Only one sample supported (Batch size of 1)."
    assert image_series.shape[:2] == keypoint_coords.shape[:2], \
        f"Image series shape is {image_series.shape} but key-points shape is {keypoint_coords.shape}"

    # Make colormap
    # indexable_cmap = cm.get_cmap('prism', keypoint_coords.shape[2])
    cm = pylab.get_cmap('gist_rainbow')

    # Permute to (N, T, H, W, C) for matplotlib
    if image_series.min() < -0.1:
        image_series = (image_series.permute(0, 1, 3, 4, 2) + 0.5).clip(0.0, 1.0).detach().cpu().numpy()
    else:
        image_series = image_series.permute(0, 1, 3, 4, 2).clip(0.0, 1.0).detach().cpu().numpy()

    # Filter for "active" key-point, i.e. key-points with an avg intensity above the threshold
    active_kp_ids = []
    for kp in range(keypoint_coords.shape[2]):
        if keypoint_coords.shape[3] == 2 or np.mean(keypoint_coords[0, :, kp, 2].cpu().numpy()) > intensity_threshold:
            active_kp_ids.append(kp)

    frame = np.zeros((W, H, C))

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    # ax.set_title('Sample + Key-Points')
    # ax.axis('tight')
    ax.axis('off')

    if save_frames:
        os.makedirs(f"{save_path}/frames/", exist_ok=True)

    """

        Initialisation

    """

    if C == 1:
        orig_im_obj = ax.imshow(frame.squeeze(), cmap='Greys', vmin=0, vmax=1)
    else:
        orig_im_obj = ax.imshow(frame)

    if key_point_trajectory:
        key_point_pos_buffer = [[] for _ in range(len(active_kp_ids))]
    line_objects = []
    orig_scatter_objects = []
    if C == 1:
        orig_im_obj.set_data(image_series[0, 0, ...].squeeze())
    else:
        orig_im_obj.set_data(image_series[0, 0, ...])

    for n_keypoint in range(len(active_kp_ids)):
        if keypoint_coords.shape[3] == 2:
            intensity = 1.0
        else:
            # intensity = keypoint_coords[0, 0, n_keypoint, 2]
            intensity = keypoint_coords[0, 0, active_kp_ids[n_keypoint], 2]
        if intensity >= 0.0:
            orig_scatter_obj = ax.scatter(0, 0, color=cm(1.*n_keypoint/len(active_kp_ids)),
                                          alpha=0.5)
            orig_scatter_objects.append(orig_scatter_obj)
            if key_point_trajectory:
                line_obj = ax.plot([0, 0], [0, 0],
                                   color=cm(1.*n_keypoint/len(active_kp_ids)),
                                   alpha=0.5)[0]
                line_objects.append(line_obj)

    """

        Iteration

    """

    def animate(t: int):
        im_frame = image_series[0, t, ...].squeeze()
        orig_im_obj.set_data(im_frame)

        for k in range(len(active_kp_ids)):
            # NOTE: The predicted keypoints are in [y(height), x(width)] coordinates
            #       in the ranges [-1.0; -1.0] to [1.0; 1.0]
            #                   ^
            #                   |
            #                   |    x[0.5, 0.5]
            #                   |
            #         ---------------------->
            #                   |
            #              x    |
            #     [-0.5, -0.5]  |
            #                   |

            img_coordinates = kpts_2_img_coordinates(keypoint_coords[0, t, active_kp_ids[k], :], (H, W))

            orig_scatter_objects[k].set_offsets([img_coordinates[0], img_coordinates[1]])

            if key_point_trajectory:
                if len(key_point_pos_buffer[k]) < trajectory_length:
                    key_point_pos_buffer[k].append([img_coordinates[0], img_coordinates[1]])
                else:
                    key_point_pos_buffer[k].pop(0)
                    key_point_pos_buffer[k].append([img_coordinates[0], img_coordinates[1]])
                combined_np_array = np.concatenate([key_point_pos_buffer[k]])
                line_objects[k].set_data(combined_np_array[:, 0], combined_np_array[:, 1])

        if save_frames:

            fig.savefig(f"{save_path}/frames/t{t}.png", bbox_inches='tight', transparent=True, pad_inches=0.0)

        return im_frame, orig_scatter_objects, line_objects

    anim = animation.FuncAnimation(fig, animate, frames=T, interval=10, repeat=False)

    # Set up formatting for the video files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20, metadata=dict(artist='me'), bitrate=1800)
    anim.save(save_path + "anim.mp4", writer=writer)

    # plt.show()
    plt.close()

    return active_kp_ids


def play_series_and_reconstruction_with_keypoints(image_series: torch.Tensor,
                                                  keypoint_coords: torch.Tensor,
                                                  feature_maps: torch.Tensor = None,
                                                  reconstructed_diff: torch.Tensor = None,
                                                  reconstruction: torch.Tensor = None,
                                                  intensity_threshold: float = 0.9,
                                                  key_point_trajectory: bool = False,
                                                  trajectory_length: int = 10, ):
    """ Visualizes the image-series tensor
        together with its reconstruction and
        given predicted key-points.

    :param image_series: Tensor of images in (N, T, C, H, W)
    :param keypoint_coords: Tensor of key-point coordinates in (N, T, K, 2/3)
    :param feature_maps: Tensor of feature maps per key-point in (N, T, K, H', W')
    :param reconstructed_diff: Tensor of predicted difference v_0 to v_t in (N, T, C, H, W)
    :param reconstruction: Tensor of predicted reconstructed image series in (N, T, C, H, w)
    :param intensity_threshold: Intensity threshold above which the key-points are plotted
    :param key_point_trajectory: Set true to also plot the trajectories of the predicted key-points
    :param trajectory_length: Length of the key-point trajectories to plot
    :return: list of ids (of keypoint_coords tensor) of "active" key-points
    """

    assert keypoint_coords.dim() == 4

    if reconstructed_diff is not None:
        assert reconstruction is None
        assert image_series.dim() == reconstructed_diff.dim() == 5
        assert image_series.shape[0:2] == keypoint_coords.shape[0:2] == reconstructed_diff.shape[0:2], \
            "Batch sizes or time-steps do not match!"
        reconstructed_image_series = reconstructed_diff + image_series[:, 0, ...]
    else:
        assert reconstruction is not None
        reconstructed_image_series = reconstruction

    (N, T, C, H, W) = tuple(image_series.shape)
    if feature_maps is not None:
        (K, Hp, Wp) = tuple(feature_maps.shape[2:])
        rgba_img_sequence = torch.zeros(size=(N, T, C + 1, H, W))
        rgba_img_sequence[:, :, :C, ...] = image_series + 0.5
        rgba_img_sequence = rgba_img_sequence.clip(0.0, 1.0).detach().cpu().numpy()
    assert C in [1, 3], "Only one or three channels supported for plotting image series!"
    assert N == 1, "Only one sample supported (Batch size of 1)."

    if key_point_trajectory:
        assert keypoint_coords.shape[1] >= trajectory_length, \
            "Use at least 'trajectory_length' time-steps to plot the key-point trajectories!"

    # Make colormap
    indexable_cmap = cm.get_cmap('prism', keypoint_coords.shape[2])

    # Permute to (N, T, H, W, C) for matplotlib
    image_series = (image_series.permute(0, 1, 3, 4, 2) + 0.5).clip(0.0, 1.0).detach().cpu().numpy()
    reconstructed_image_series = (reconstructed_image_series.permute(0, 1, 3, 4, 2) + 0.5).clip(0.0,
                                                                                                1.0).detach().cpu().numpy()

    # Filter for "active" key-point, i.e. key-points with an avg intensity above the threshold
    active_kp_ids = []
    for kp in range(keypoint_coords.shape[2]):
        if keypoint_coords.shape[3] == 2 or np.mean(keypoint_coords[0, :, kp, 2].cpu().numpy()) > intensity_threshold:
            active_kp_ids.append(kp)

    frame = np.zeros((W, H, C))

    if feature_maps is not None:
        rgba_frame = np.zeros((W, H, C))

        fig, ax = plt.subplots(1, 3, figsize=(15, 10))
        ax[0].set_title('Sample + Key-Points')
        ax[1].set_title('Sample + Feature-Maps')
        ax[2].set_title('Reconstruction')
    else:
        fig, ax = plt.subplots(1, 2, figsize=(15, 10))
        ax[0].set_title('Sample + Key-Points')
        ax[-1].set_title('Reconstruction')

    """
    
        Initialisation
    
    """

    if C == 1:
        orig_im_obj = ax[0].imshow(frame.squeeze(), cmap='Greys', vmin=0, vmax=1)
        if feature_maps is not None:
            rgba_im_obj = ax[1].imshow(rgba_frame.squeeze(), cmap='Greys', vmin=0, vmax=1)
        rec_im_obj = ax[-1].imshow(frame.squeeze(), cmap='Greys', vmin=0, vmax=1)
    else:
        orig_im_obj = ax[0].imshow(frame)
        if feature_maps is not None:
            rgba_im_obj = ax[1].imshow(rgba_frame)
        rec_im_obj = ax[-1].imshow(frame)

    if key_point_trajectory:
        key_point_pos_buffer = [[] for _ in range(len(active_kp_ids))]
    line_objects = []
    orig_scatter_objects = []
    if C == 1:
        orig_im_obj.set_data(image_series[0, 0, ...].squeeze())
        if feature_maps is not None:
            rgba_im_obj.set_data(rgba_img_sequence[0, 0, ...].squeeze())
        rec_im_obj.set_data(reconstructed_image_series[0, 0, ...].squeeze())
    else:
        orig_im_obj.set_data(image_series[0, 0, ...])
        if feature_maps is not None:
            rgba_im_obj.set_data(rgba_img_sequence[0, 0, ...].transpose(1, 2, 0))
        rec_im_obj.set_data(reconstructed_image_series[0, 0, ...])
    for n_keypoint in range(len(active_kp_ids)):
        if keypoint_coords.shape[3] == 2:
            intensity = 1.0
        else:
            # intensity = keypoint_coords[0, 0, n_keypoint, 2]
            intensity = keypoint_coords[0, 0, active_kp_ids[n_keypoint], 2]
        if intensity >= 0.0:
            orig_scatter_obj = ax[0].scatter(0, 0, color=indexable_cmap(n_keypoint / len(active_kp_ids)),
                                             alpha=0.5)
            orig_scatter_objects.append(orig_scatter_obj)
            if key_point_trajectory:
                line_obj = ax[0].plot([0, 0], [0, 0],
                                      color=indexable_cmap(n_keypoint / len(active_kp_ids)),
                                      alpha=0.5)[0]
                line_objects.append(line_obj)

    """
    
        Iteration
    
    """

    def animate(t: int):
        im_frame = image_series[0, t, ...].squeeze()
        if feature_maps is not None:
            rgba_frame = rgba_img_sequence[0, t, ...].transpose(1, 2, 0).squeeze().copy()
        orig_im_obj.set_data(im_frame)
        rec_frame = reconstructed_image_series[0, t, ...].squeeze()
        rec_im_obj.set_data(rec_frame)

        for k in range(len(active_kp_ids)):
            # NOTE: The predicted keypoints are in [y(height), x(width)] coordinates
            #       in the ranges [-1.0; -1.0] to [1.0; 1.0]
            #                   ^
            #                   |
            #                   |    x[0.5, 0.5]
            #                   |
            #         ---------------------->
            #                   |
            #              x    |
            #     [-0.5, -0.5]  |
            #                   |

            x_coord = int((-keypoint_coords[0, t, active_kp_ids[k], 1] + 1.0) / 2 * W)
            y_coord = int((keypoint_coords[0, t, active_kp_ids[k], 0] + 1.0) / 2 * H)

            orig_scatter_objects[k].set_offsets([x_coord, y_coord])

            if key_point_trajectory:
                if len(key_point_pos_buffer[k]) < trajectory_length:
                    key_point_pos_buffer[k].append([x_coord, y_coord])
                else:
                    key_point_pos_buffer[k].pop(0)
                    key_point_pos_buffer[k].append([x_coord, y_coord])
                combined_np_array = np.concatenate([key_point_pos_buffer[k]])
                line_objects[k].set_data(combined_np_array[:, 0], combined_np_array[:, 1])

            if feature_maps is not None:
                upscaled_feature_map = F.interpolate(feature_maps[0, t:t + 1, k:k + 1, ...], size=(H, W)).cpu().numpy()
                rgba_frame[..., 3] += upscaled_feature_map.squeeze()

        if feature_maps is not None:
            rgba_frame = rgba_frame.clip(0.0, 1.0)
            rgba_im_obj.set_data(rgba_frame)
            return im_frame, rec_frame, orig_scatter_objects, line_objects, rgba_im_obj

        else:
            return im_frame, rec_frame, orig_scatter_objects, line_objects, None

    anim = animation.FuncAnimation(fig, animate, frames=T, interval=10, repeat=False)

    # Set up formatting for the video files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20, metadata=dict(artist='me'), bitrate=1800)
    anim.save('anim.mp4', writer=writer)

    plt.show()

    return active_kp_ids


def gen_eval_imgs(sample: torch.Tensor,
                  reconstruction: torch.Tensor,
                  key_points: torch.Tensor,
                  intensity_threshold: float = 0.75,
                  key_point_trajectory: bool = False):
    """ Creates an torch image series tensor out of a series of pyplot images,
        that show the sample at time 0, at time n, the difference and its prediction.

        (Only for the first sample of the mini-batch.)

    :param sample: Torch tensor of sample image sequence in (N, T, C, H, W)
    :param reconstruction: Torch tensor of reconstructed image series (N, T, C, H, W)
    :param key_points: Torch tensor of key-point coordinates in (N, T, C, 3) or (N, T, C, 2)
    :param key_point_trajectory: Set true to include trajectories of key-points into the plot
    :return: Series of images as torch tensor in (1, T, C, H, W) in range [0, 1]
    """
    assert sample.ndim == 5
    assert reconstruction.ndim == 5
    if sample.min() < -0.01:
        sample = (sample + 0.5).clip(0.0, 1.0)
        reconstruction = (reconstruction + 0.5).clip(0.0, 1.0)
    # print(sample.min())
    # print(sample.max())
    assert key_points.ndim == 4
    # assert key_points.shape[3] in [2, 3]
    assert key_points[..., :2].min() >= -1.0
    assert key_points[..., :2].max() <= 1.0
    if key_points.shape[3] == 3:
        assert key_points[..., 2].min() >= 0
        assert key_points[..., 2].max() <= 1.0
    if key_point_trajectory:
        assert key_points.shape[1] >= 5, "Use at least 5 time-steps to plot the key-point trajectories!"
        position_buffer = []

    cm = pylab.get_cmap('gist_rainbow')

    torch_img_series_tensor = None
    for t in range(sample.shape[1]):
        fig, ax = plt.subplots(1, 5, figsize=(15, 4))
        # viridis = cm.get_cmap('viridis', key_points.shape[2])

        #
        # Future time-steps and key-points
        #

        if key_point_trajectory:
            if len(position_buffer) < 5:
                position_buffer.append(key_points[0, t, ...].cpu().numpy())
            else:
                position_buffer.pop(0)
                position_buffer.append(key_points[0, t, ...].cpu().numpy())
            assert len(position_buffer) <= 5

        ax[0].imshow(sample[0, t, ...].permute(1, 2, 0).cpu().numpy())
        for kp in range(key_points.shape[2]):
            if key_points.shape[-1] == 2 or key_points[0, t, kp, 2] > intensity_threshold:
                # NOTE: pyplot.scatter uses x as height and y as width, with the origin in the top left
                x_coord = int(((-key_points[0, t, kp, 1] + 1.0) / 2.0) * sample.shape[4])  # Width
                y_coord = int(((key_points[0, t, kp, 0] + 1.0) / 2.0) * sample.shape[3])  # Height
                ax[0].scatter(x_coord, y_coord, color=cm(1.*kp/key_points.shape[2]), marker="^", s=50, alpha=0.9)
                if key_point_trajectory:
                    ax[0].plot(x=[position_buffer[t][kp, 0] for t in range(len(position_buffer))],
                               y=[position_buffer[t][kp, 1] for t in range(len(position_buffer))],
                               color='yellow')
        ax[0].set_title(f'sample t{t}')

        #
        #   Initial time-step
        #

        ax[1].imshow(sample[0, 0, ...].permute(1, 2, 0).cpu().numpy())
        ax[1].scatter(int(0.1 * sample.shape[4]), int(0.8 * sample.shape[3]), marker='x', color='red')
        ax[1].scatter(int(0.3 * sample.shape[4]), int(0.8 * sample.shape[3]), marker='x', color='white')
        ax[1].set_title('sample t0')

        #
        #   Target diff.
        #

        ax[2].imshow(((sample[0, t, ...] - sample[0, 0, ...]) + 0.5).clip(0.0, 1.0).permute(1, 2, 0).cpu().numpy())
        ax[2].set_title('target difference')

        #
        #   Predicted diff.
        #

        ax[3].imshow(
            ((reconstruction[0, t, ...] - sample[0, 0, ...]) + 0.5).clip(0.0, 1.0).permute(1, 2, 0).cpu().numpy())
        ax[3].set_title('predicted difference')

        #
        #   Reconstruction
        #

        ax[4].imshow(reconstruction[0, t, ...].permute(1, 2, 0).cpu().numpy())
        ax[4].set_title('reconstruction')

        memory_buffer = io.BytesIO()
        plt.savefig(memory_buffer, format='png')
        memory_buffer.seek(0)
        pil_img = Image.open(memory_buffer)
        pil_to_tensor = transforms.ToTensor()(pil_img).unsqueeze(0)
        plt.close()
        memory_buffer.close()

        if torch_img_series_tensor is None:
            torch_img_series_tensor = pil_to_tensor
        else:
            torch_img_series_tensor = torch.cat([torch_img_series_tensor, pil_to_tensor])

    assert torch_img_series_tensor.ndim == 4
    return torch_img_series_tensor[:, :3, ...].unsqueeze(0)


def plot_keypoint_amplitudes(keypoint_coordinates: torch.Tensor,
                             target_path: str,
                             intensity_threshold: float = 0.5):
    """ Plots the amplitudes of the x- and y-coordinates separately for each
        active key-point.

    :param keypoint_coordinates: Torch tensor of key-point coordinates in (1, T, K, 2/3)
    :param target_path: Path to save plot to.
    :param intensity_threshold: Threshold for avg. intensity to define a keypoint as active (In range [0, 1]).
        If key-points do not have intensity values, e.g. their last dim is of size 2,
        then they are all assumed as active.
    :return:
    """
    assert os.path.isdir(target_path)
    assert keypoint_coordinates.dim() == 4
    assert keypoint_coordinates.shape[-1] in [2, 3]
    assert 0.0 <= intensity_threshold <= 1.0

    T = keypoint_coordinates.shape[1]
    cm = pylab.get_cmap('gist_rainbow')

    fig, ax = plt.subplots(3, 1, figsize=(10, 15))
    if keypoint_coordinates.shape[-1] == 3:
        ax[0].set_title(f"x coordinate (mean. int > {intensity_threshold})")
        ax[1].set_title(f"y coordinate (mean. int > {intensity_threshold})")
    else:
        ax[0].set_title(f"x coordinate")
        ax[1].set_title(f"y coordinate")
    ax[2].set_title("intensity")
    for n_keypoint in range(keypoint_coordinates.shape[2]):

        if keypoint_coordinates.shape[-1] == 2:
            mean_int = 1.0
        else:
            mean_int = np.mean(keypoint_coordinates[0, :, n_keypoint, 2].cpu().numpy().squeeze())

        if mean_int >= intensity_threshold:
            ax[0].plot(np.arange(0, T),
                       keypoint_coordinates[0:1, :, n_keypoint, 0].cpu().numpy().squeeze(),
                       color=cm(1.*n_keypoint/keypoint_coordinates.shape[2]))
            ax[1].plot(np.arange(0, T),
                       keypoint_coordinates[0:1, :, n_keypoint, 1].cpu().numpy().squeeze(),
                       color=cm(1.*n_keypoint/keypoint_coordinates.shape[2]))
        if keypoint_coordinates.shape[3] == 3:
            ax[2].plot(np.arange(0, T),
                       keypoint_coordinates[0:1, :, n_keypoint, 2].cpu().numpy().squeeze(),
                       color=cm(1.*n_keypoint/keypoint_coordinates.shape[2]))
        else:
            ax[2].plot(np.arange(0, T),
                       np.array([1] * T),
                       color=cm(1.*n_keypoint/keypoint_coordinates.shape[2]))
    plt.savefig(f'{target_path}/kp_amps.png')
    plt.close()


def imprint_img_with_kpts(img: torch.Tensor, kpt: torch.Tensor) -> np.ndarray:
    """ Makes a matplotlib plot from the given image,
        with scatter-plots at the key-point positions.
        Then converts the result back to an image tensor.

    :param img: Image tensor in (1, 1, C, H, W)
    :param kpt: Key-point coordinates in (1, 1, K, 2/3) in video-structure format:
                               ^
                               |
                               |    x[0.5, 0.5]
                               |
                     ---------------------->
                               |
                          x    |
                 [-0.5, -0.5]  |


    :return:
    """

    H, W = img.shape[-2:]

    dpi = mpl.rcParams['figure.dpi']

    figsize = (H / dpi, W / dpi)

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    fig.patch.set_facecolor('xkcd:mint green')

    canvas = FigureCanvas(fig)
    ax.axis('off')
    ax.imshow(img.squeeze().permute(1, 2, 0).detach().cpu())
    for k in range(kpt.shape[2]):
        kpt_w = (kpt.squeeze()[..., k, 0] + 1) / 2 * img.shape[-1]  # W
        kpt_h = -(kpt.squeeze()[..., k, 1] + 1) / 2 * img.shape[-2]  # H
        print(f"{kpt_w}, {kpt_h}")
        ax.scatter(kpt_w, kpt_h, s=50, marker='x', color='black')
    fig_width, fig_height = fig.get_size_inches() * fig.get_dpi()
    plt.show()
    exit()
    canvas.draw()
    image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape((int(fig_height), int(fig_width), 3))
    return image

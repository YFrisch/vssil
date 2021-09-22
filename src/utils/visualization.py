import io
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, animation
from PIL import Image
from torchvision import transforms
from cv2 import VideoWriter, VideoWriter_fourcc,\
    normalize, NORM_MINMAX, CV_32F


def numpy_to_mp4(img_array: np.ndarray, target_path: str = 'test.avi'):
    """ Takes a numpy array in (T, H, W, C) format and makes a video out of it."""
    width = 64
    height = 64
    fps = 16
    # Convert to 255 RGB
    norm_img_array = normalize(img_array, None, alpha=0, beta=255,
                               norm_type=NORM_MINMAX, dtype=CV_32F)
    norm_img_array = norm_img_array.astype(np.uint8)
    assert img_array.shape[0] % fps == 0
    sec = int(img_array.shape[0]/fps)
    fourcc = VideoWriter_fourcc(*'MPEG')
    video = VideoWriter(target_path, fourcc, float(fps), (width, height))
    for frame_count in range(fps*sec):
        video.write(norm_img_array[frame_count, ...])
    video.release()


def play_series_and_reconstruction_with_keypoints(image_series: torch.Tensor,
                                                  keypoint_coords: torch.Tensor,
                                                  reconstructed_diff: torch.Tensor = None,
                                                  reconstruction: torch.Tensor = None,
                                                  intensity_threshold: float = 0.9,
                                                  key_point_trajectory: bool = False,
                                                  trajectory_length: int = 10,):
    """ Visualizes the an image-series tensor
        together with its reconstruction and
        given predicted key-points.

    :param image_series: Tensor of images in (N, T, C, H, W)
    :param keypoint_coords: Tensor of key-point coordinates in (N, T, K, 2/3)
    :param reconstructed_diff: Tensor of predicted difference v_0 to v_t in (N, T, C, H, W)
    :param reconstruction: Tensor of predicted reconstructed image series in (N, T, C, H, w)
    :param intensity_threshold: Intensity threshold above which the key-points are plotted
    :param key_point_trajectory: Set true to also plot the trajectories of the predicted key-points
    :param trajectory_length: Length of the key-point trajectories to plot
    :return:
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
    assert C in [1, 3], "Only one or three channels supported for plotting image series!"
    assert N == 1, "Only one sample supported (Batch size of 1)."

    if key_point_trajectory:
        assert keypoint_coords.shape[1] >= trajectory_length,\
            "Use at least 'trajectory_length' time-steps to plot the key-point trajectories!"

    # Make colormap
    indexable_cmap = cm.get_cmap('prism', keypoint_coords.shape[1])

    # Permute to (N, T, H, W, C) for matplotlib
    image_series = (image_series.permute(0, 1, 3, 4, 2) + 0.5).clip(0.0, 1.0).detach().cpu().numpy()
    reconstructed_image_series = (reconstructed_image_series.permute(0, 1, 3, 4, 2) + 0.5).clip(0.0, 1.0).detach().cpu().numpy()

    frame = np.zeros((W, H, C))

    fig, ax = plt.subplots(1, 2)
    ax[0].set_title('Sample + Key-Points')
    ax[1].set_title('Reconstruction')
    if C == 1:
        orig_im_obj = ax[0].imshow(frame.squeeze(), cmap='Greys', vmin=0, vmax=1)
        rec_im_obj = ax[1].imshow(frame.squeeze(), cmap='Greys', vmin=0, vmax=1)
    else:
        orig_im_obj = ax[0].imshow(frame)
        rec_im_obj = ax[1].imshow(frame)

    if key_point_trajectory:
        key_point_pos_buffer = [[] for _ in range(keypoint_coords.shape[2])]
    line_objects = []
    orig_scatter_objects = []
    if C == 1:
        orig_im_obj.set_data(image_series[0, 0, ...].squeeze())
        rec_im_obj.set_data(reconstructed_image_series[0, 0, ...].squeeze())
    else:
        orig_im_obj.set_data(image_series[0, 0, ...])
        rec_im_obj.set_data(reconstructed_image_series[0, 0, ...])
    for n_keypoint in range(keypoint_coords.shape[2]):
        if keypoint_coords.shape[3] == 2:
            intensity = 1.0
        else:
            intensity = keypoint_coords[0, 0, n_keypoint, 2]
        if intensity > 0.0:
            #orig_scatter_obj = ax[0].scatter(0, 0, cmap=viridis, alpha=0.5)
            orig_scatter_obj = ax[0].scatter(0, 0, color=indexable_cmap(n_keypoint/keypoint_coords.shape[2]), alpha=0.5)
            orig_scatter_objects.append(orig_scatter_obj)
            if key_point_trajectory:
                #line_obj = ax[0].plot([0, 0], [0, 0], color='yellow', alpha=0.25)[0]
                line_obj = ax[0].plot([0, 0], [0, 0],
                                      #color=viridis.colors[n_keypoint],
                                      color=indexable_cmap(n_keypoint/keypoint_coords.shape[2]),
                                      alpha=0.5)[0]
                line_objects.append(line_obj)

    def animate(t: int):
        im_frame = image_series[0, t, ...].squeeze()
        orig_im_obj.set_data(im_frame)
        rec_frame = reconstructed_image_series[0, t, ...].squeeze()
        rec_im_obj.set_data(rec_frame)
        for n_keypoint in range(keypoint_coords.shape[2]):
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
            # x1 = int((keypoint_coords[0, t, n_keypoint, 0] + 1.0)/2 * W)
            # x2 = int((-keypoint_coords[0, t, n_keypoint, 1] + 1.0)/2 * H)
            x_coord = int((-keypoint_coords[0, t, n_keypoint, 1] + 1.0) / 2 * W)
            y_coord = int((keypoint_coords[0, t, n_keypoint, 0] + 1.0)/2 * H)

            if keypoint_coords.shape[3] == 2:
                intensity = 1.0
            else:
                intensity = keypoint_coords[0, 0, n_keypoint, 2]
            if intensity > intensity_threshold:
                orig_scatter_objects[n_keypoint].set_offsets([x_coord, y_coord])

                if key_point_trajectory:
                    if len(key_point_pos_buffer[n_keypoint]) < trajectory_length:
                            key_point_pos_buffer[n_keypoint].append([x_coord, y_coord])
                    else:
                        key_point_pos_buffer[n_keypoint].pop(0)
                        key_point_pos_buffer[n_keypoint].append([x_coord, y_coord])
                    combined_np_array = np.concatenate([key_point_pos_buffer[n_keypoint]])
                    #assert combined_np_array.shape == (trajectory_length, 2),\
                    #    f'{combined_np_array.shape} != ({trajectory_length}, 2)'
                    line_objects[n_keypoint].set_data(combined_np_array[:, 0], combined_np_array[:, 1])
            else:
                pass
                # orig_scatter_objects[n_keypoint].set_offsets([0.0, 0.0])

        return im_frame, rec_frame, orig_scatter_objects, line_objects

    anim = animation.FuncAnimation(fig, animate, frames=T, interval=10, repeat=False)

    # Set up formatting for the video files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=20, metadata=dict(artist='me'), bitrate=1800)
    anim.save('anim.mp4', writer=writer)

    plt.show()


def gen_eval_imgs(sample: torch.Tensor,
                  reconstruction: torch.Tensor,
                  key_points: torch.Tensor,
                  key_point_trajectory: bool = False):
    """ Creates an torch image series tensor out of a series of pyplot images,
        that show the sample at time 0, at time n, the difference and its prediction.

        (Only for the first sample of the mini-batch.)

    :param sample: Torch tensor of sample image sequence in (N, T, C, H, W)
    :param reconstruction: Torch tensor of reconstructed image series (N, T, C, H, W)
    :param key_points: Torch tensor of key-point coordinates in (N, T, C, 3) or (N, T, C, 2)
    :return: Series of images as torch tensor in (1, T, C, H, W) in range [0, 1]
    """
    assert sample.ndim == 5
    assert reconstruction.ndim == 5
    sample = sample + 0.5
    reconstruction = reconstruction + 0.5
    assert key_points.ndim == 4
    assert key_points.shape[3] in [2, 3]
    assert key_points[..., :2].min() >= -1.0
    assert key_points[..., :2].max() <= 1.0
    if key_points.shape[3] == 3:
        assert key_points[..., 2].min() >= 0
        assert key_points[..., 2].max() <= 1.0
    if key_point_trajectory:
        assert key_points.shape[1] >= 5, "Use at least 5 time-steps to plot the key-point trajectories!"
        position_buffer = []

    torch_img_series_tensor = None
    for t in range(sample.shape[1]):
        fig, ax = plt.subplots(1, 5, figsize=(15, 4))
        viridis = cm.get_cmap('viridis', key_points.shape[2])

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
            if key_points.shape[-1] == 2 or key_points[0, t, kp, 2] > 0.75:
                # NOTE: pyplot.scatter uses x as height and y as width, with the origin in the top left
                x_coord = int(((-key_points[0, t, kp, 1] + 1.0) / 2.0) * sample.shape[4])  # Width
                y_coord = int(((key_points[0, t, kp, 0] + 1.0) / 2.0) * sample.shape[3])  # Height
                ax[0].scatter(x_coord, y_coord, marker='o', cmap=viridis)
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

        ax[2].imshow(((sample[0, t, ...]-sample[0, 0, ...]) + 0.5).clip(0.0, 1.0).permute(1, 2, 0).cpu().numpy())
        ax[2].set_title('target difference')

        #
        #   Predicted diff.
        #

        ax[3].imshow(((reconstruction[0, t, ...] - sample[0, 0, ...]) + 0.5).clip(0.0, 1.0).permute(1, 2, 0).cpu().numpy())
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

        #del pil_img, pil_to_tensor, memory_buffer, fig, ax, viridis, x_coord, y_coord

    #del sample, reconstruction, key_points
    assert torch_img_series_tensor.ndim == 4
    return torch_img_series_tensor[:, :3, ...].unsqueeze(0)

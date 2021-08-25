import io
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm, animation
from PIL import Image
from torchvision import transforms


def play_series_and_reconstruction_with_keypoints(image_series: torch.Tensor,
                                                  keypoint_coords: torch.Tensor,
                                                  reconstructed_diff: torch.Tensor = None,
                                                  reconstruction: torch.Tensor = None):
    """ Visualizes the an image-series tensor
        together with its reconstruction and
        given predicted key-points.

        TODO: Check why key-point colors are still inconsistent
        TODO: Check if scatter plots of each time-steps are actually deleted in the next time-step?

    :param image_series: Tensor of images in (N, T, C, H, W)
    :param keypoint_coords: Tensor of key-point coordinates in (N, T, K, 2/3)
    :param reconstructed_diff: Tensor of predicted difference v_0 to v_t in (N, T, C, H, W)
    :param reconstruction: Tensor of predicted reconstructed image series in (N, T, C, H, w)
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

    # Make colormap
    viridis = cm.get_cmap('viridis', keypoint_coords.shape[1])

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

    orig_scatter_objects = []
    rec_scatter_objects = []
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
            orig_scatter_obj = ax[0].scatter(0, 0, cmap=viridis, alpha=0.5)
            orig_scatter_objects.append(orig_scatter_obj)

    def animate(t: int):
        im_frame = image_series[0, t, ...].squeeze()
        orig_im_obj.set_data(im_frame)
        rec_frame = reconstructed_image_series[0, t, ...].squeeze()
        rec_im_obj.set_data(rec_frame)
        for n_keypoint in range(keypoint_coords.shape[2]):
            # NOTE: The predicted keypoints are in [y(width), x(height)] coordinates
            #       in the ranges [-1.0; -1.0] to [1.0; 1.0]
            x1 = int((keypoint_coords[0, t, n_keypoint, 0] + 1.0)/2 * W)
            x2 = int((-keypoint_coords[0, t, n_keypoint, 1] + 1.0)/2 * H)

            if keypoint_coords.shape[3] == 2:
                intensity = 1.0
            else:
                intensity = keypoint_coords[0, 0, n_keypoint, 2]
            if intensity > 0.9:
                orig_scatter_objects[n_keypoint].set_offsets([x1, x2])
            else:
                orig_scatter_objects[n_keypoint].set_offsets([0.0, 0.0])

        return im_frame, rec_frame, orig_scatter_objects, rec_scatter_objects  # , rec_diff_frame

    anim = animation.FuncAnimation(fig, animate, frames=T, interval=10, repeat=False)

    # Set up formatting for the video files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    anim.save('anim.mp4', writer=writer)

    plt.show()


def gen_eval_imgs(sample: torch.Tensor,
                  reconstructed_diff: torch.Tensor,
                  key_points: torch.Tensor):
    """ Creates an torch image series tensor out of a series of pyplot images,
        that show the sample at time 0, at time n, the difference and its prediction.

        (Only for the first sample of the mini-batch.)

    :param sample: Torch tensor of sample image sequence in (N, T, C, H, W)
    :param reconstructed_diff: Torch tensor of predicted differences from t0 to tT in (N, T, C, H, W)
    :param key_points: Torch tensor of key-point coordinates in (N, T, C, 3) or (N, T, C, 2)
    :return: Series of images as torch tensor in (1, T, C, H, W) in range [0, 1]
    """
    assert sample.ndim == 5
    sample = sample + 0.5
    assert reconstructed_diff.ndim == 5
    assert key_points.ndim == 4
    assert key_points.shape[3] in [2, 3]

    torch_img_series_tensor = None
    for t in range(sample.shape[1]):
        fig, ax = plt.subplots(1, 5, figsize=(15, 4))
        viridis = cm.get_cmap('viridis', key_points.shape[2])
        ax[0].imshow(sample[0, t, ...].permute(1, 2, 0).cpu().numpy())
        for kp in range(key_points.shape[2]):
            if key_points.shape[-1] == 2 or key_points[0, t, kp, 2] > 0.75:
                # NOTE: pyplot.scatter uses x as height and y as width, with the origin in the top left
                key_point_x1 = int(((key_points[0, t, kp, 0] + 1.0) / 2.0) * sample.shape[4])  # Width
                key_point_x2 = int(((-key_points[0, t, kp, 1] + 1.0) / 2.0) * sample.shape[3])  # Height
                ax[0].scatter(key_point_x1, key_point_x2, marker='o', cmap=viridis)
        ax[0].set_title(f'sample t{t}')
        ax[1].imshow(sample[0, 0, ...].permute(1, 2, 0).cpu().numpy())
        ax[1].set_title('sample t0')
        ax[2].imshow(((sample[0, t, ...]-sample[0, 0, ...]) + 0.5).clip(0.0, 1.0).permute(1, 2, 0).cpu().numpy())
        ax[2].set_title('target difference')
        ax[3].imshow((reconstructed_diff[0, t, ...] + 0.5).clip(0.0, 1.0).permute(1, 2, 0).cpu().numpy())
        ax[3].set_title('predicted difference')
        ax[4].imshow((reconstructed_diff[0, t, ...] + sample[0, 0, ...]).clip(0.0, 1.0).permute(1, 2, 0).cpu().numpy())
        ax[4].set_title('reconstruction')

        memory_buffer = io.BytesIO()
        plt.savefig(memory_buffer, format='png')
        memory_buffer.seek(0)
        pil_img = Image.open(memory_buffer)
        pil_to_tensor = transforms.ToTensor()(pil_img).unsqueeze(0)
        plt.close()

        if torch_img_series_tensor is None:
            torch_img_series_tensor = pil_to_tensor
        else:
            torch_img_series_tensor = torch.cat([torch_img_series_tensor, pil_to_tensor])

    assert torch_img_series_tensor.ndim == 4
    return torch_img_series_tensor[:, :3, ...].unsqueeze(0)

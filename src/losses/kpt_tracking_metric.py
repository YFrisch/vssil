import torch

from kornia.filters import sobel
from torch.nn.functional import kl_div
from src.utils.kpt_utils import get_image_patches

from .utils import differentiable_histogram


def kpt_tracking_metric(kpt_sequence: torch.Tensor,
                        img_sequence: torch.Tensor,
                        patch_size: tuple,
                        n_bins: int = 30,
                        p: float = float('inf')) -> torch.Tensor:
    """ Evaluates the consistency of tracked objects of key-points.

        For this, the color and gradient histograms of image patches around key-points
        are compared (max-norm difference) across time.

        The gradients for each channel are obtained using a sobel filter.

    :param kpt_sequence: Torch tensor of key-point coordinates in (N, T, K, D)
    :param img_sequence: Torch tensor of sequential frames in (N, T, C, H, W)
    :param patch_size: Size of the patches to compare (H', W')
    :param n_bins: Number histogram bins
    :param p: Order of norm distance used to compare color / gradient distributions
    :return:
    """

    N, T, K, D = kpt_sequence.shape
    _, _, C, H, W = img_sequence.shape
    Hp, Wp = patch_size

    patches = get_image_patches(img_sequence, kpt_sequence, patch_size)  # (N, T, K, C, Hp, Wp)

    grads = sobel(patches.view((N * T * K, C, Hp, Wp))).view((N, T, K, C, Hp, Wp))

    color_hists = torch.empty((N, T, K, C, n_bins))
    grad_hists = torch.empty((N, T, K, C, n_bins))

    # Compute histograms
    for n in range(N):
        for t in range(T):
            for k in range(K):
                for c in range(C):
                    color_hists[n, t, k, c] = torch.histc(patches[n, t, k, c], bins=n_bins)
                    color_hists[n, t, k, c] /= torch.sum(color_hists[n, t, k, c], dim=-1)
                    #color_hists[n, t, k, c] = differentiable_histogram(
                    #    patches[n, t, k, c], bins=n_bins, min=0.0, max=1.0
                    #)

                    grad_hists[n, t, k, c] = torch.histc(grads[n, t, k, c], bins=n_bins)
                    grad_hists[n, t, k, c] /= torch.sum(grad_hists[n, t, k, c], dim=-1)
                    #grad_hists[n, t, k, c] = differentiable_histogram(
                    #    grads[n, t, k, c], bins=n_bins, min=0.0, max=1.0
                    #)

    # color_hists /= (Hp * Wp)
    # grad_hists /= (Hp * Wp)

    joint_hists = torch.multiply(color_hists, grad_hists)

    # Get distances
    color_dist = torch.empty(N, K, T - 1)
    grad_dist = torch.empty(N, K, T - 1)
    joint_dist = torch.empty(N, K, T - 1)

    for k in range(K):
        for t in range(T - 1):
            if p == 'dkl':
                color_dist[:, k, t] = kl_div(color_hists[:, t:t + 1, k, :], color_hists[:, t + 1:t + 2, k, :],
                                             reduction='batchmean')
                grad_dist[:, k, t] = kl_div(grad_hists[:, t:t + 1, k, :], grad_hists[:, t + 1:t + 2, k, :],
                                            reduction='batchmean')
                joint_dist[:, k, t] = kl_div(joint_hists[:, t:t + 1, k, :], joint_hists[:, t + 1:t + 2, k, :],
                                             reduction='batchmean')
            else:
                color_dist[:, k, t] = torch.norm(color_hists[:, t:t + 1, k, :] - color_hists[:, t + 1:t + 2, k, :], p=p)
                grad_dist[:, k, t] = torch.norm(grad_hists[:, t:t + 1, k, :] - grad_hists[:, t + 1:t + 2, k, :], p=p)
                joint_dist[:, k, t] = torch.norm(joint_hists[:, t:t + 1, k, :] - joint_hists[:, t + 1:t + 2, k, :], p=p)

    # return color_dist.mean() * grad_dist.mean()
    return joint_dist.mean(), color_dist.mean(), grad_dist.mean()

import torch

from kornia.filters import sobel
from torch.nn.functional import kl_div
from src.utils.kpt_utils import get_image_patches

from .utils import differentiable_histogram


def kpt_visual_metric(kpt_sequence: torch.Tensor,
                      img_sequence: torch.Tensor,
                      patch_size: tuple,
                      n_bins: int = 30,
                      p: float = float('inf')) -> torch.Tensor:
    """ Evaluates the visual difference between the different objects tracked among key-points.



    :param kpt_sequence:
    :param img_sequence:
    :param patch_size:
    :param n_bins:
    :param p:
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
                    # color_hists[n, t, k, c] = torch.histc(patches[n, t, k, c], bins=n_bins)
                    # color_hists[n, t, k, c] /= torch.sum(color_hists[n, t, k, c], dim=-1)
                    color_hists[n, t, k, c] = differentiable_histogram(
                        patches[n, t, k, c], bins=n_bins, min=0.0, max=1.0
                    )

                    # grad_hists[n, t, k, c] = torch.histc(grads[n, t, k, c], bins=n_bins)
                    # grad_hists[n, t, k, c] /= torch.sum(grad_hists[n, t, k, c], dim=-1)
                    grad_hists[n, t, k, c] = differentiable_histogram(
                        grads[n, t, k, c], bins=n_bins, min=0.0, max=1.0
                    )

    joint_hists = torch.multiply(color_hists, grad_hists)

    """
    #color_dists = torch.norm(color_hists.unsqueeze(2) - color_hists.unsqueeze(3), dim=[-1], p=p)   # (N, T, K, K)
    #grad_dists = torch.norm(grad_hists.unsqueeze(2) - grad_hists.unsqueeze(3), dim=[-1], p=p)  # (N, T, K, K)

    #color_dists = torch.sum(color_dists, dim=[-1, -2])/(K * (K-1))  # (N, T)
    #grad_dists = torch.sum(grad_dists, dim=[-1, -2])/(K * (K-1))  # (N, T)

    if p == 'dkl':
        joint_dists = kl_div(joint_hists.unsqueeze(2), joint_hists.unsqueeze(3))
        print(joint_dists.shape)
    else:
        joint_dists = torch.norm(joint_hists.unsqueeze(2) - joint_hists.unsqueeze(3), dim=[-1], p=p)  # (N, T, K, K)
    joint_dists = torch.sum(joint_dists, dim=[-1, -2])/(K * (K-1))  # (N, T)
    # joint_dists = torch.sum(joint_dists, dim=[-3]) / (K * (K - 1))  # (N, K, K)
    """

    joint_dist = torch.zeros([N, T, K, K])
    color_dist = torch.zeros([N, T, K, K])
    grad_dist = torch.zeros([N, T, K, K])

    for n in range(N):
        for t in range(T):
            for k in range(K):
                for _k in range(K):
                    if p == 'dkl':
                        joint_dist[n, t, k, _k] = kl_div(joint_hists[n, t, k, :], joint_hists[n, t, _k, :],
                                                         reduction='batchmean')
                        color_dist[n, t, k, _k] = kl_div(color_hists[n, t, k, :], color_hists[n, t, _k, :],
                                                         reduction='batchmean')
                        grad_dist[n, t, k, _k] = kl_div(grad_hists[n, t, k, :], grad_hists[n, t, _k, :],
                                                        reduction='batchmean')
                    else:
                        joint_dist[n, t, k, _k] = torch.norm(joint_hists[n, t, k, :] - joint_hists[n, t, _k, :], p=p)
                        color_dist[n, t, k, _k] = torch.norm(color_hists[n, t, k, :] - color_hists[n, t, _k, :], p=p)
                        grad_dist[n, t, k, _k] = torch.norm(grad_hists[n, t, k, :] - grad_hists[n, t, _k, :], p=p)

    joint_dist = torch.sum(joint_dist, dim=[-2, -1]) / (K * (K - 1))
    color_dist = torch.sum(color_dist, dim=[-2, -1]) / (K * (K - 1))
    grad_dist = torch.sum(grad_dist, dim=[-2, -1]) / (K * (K - 1))

    # return color_dists.mean() * grad_dists.mean()
    # return joint_dists.mean(), None, None #, color_dists.mean(), grad_dists.mean()

    return joint_dist.mean(), color_dist.mean(), grad_dist.mean()

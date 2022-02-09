import torch

from kornia.filters import sobel
from src.utils.kpt_utils import get_image_patches


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
                    color_hists[n, t, k, c] = torch.histc(patches[n, t, k, c], bins=n_bins)
                    color_hists[n, t, k, c] /= torch.sum(color_hists[n, t, k, c], dim=-1)

                    grad_hists[n, t, k, c] = torch.histc(grads[n, t, k, c], bins=n_bins)
                    grad_hists[n, t, k, c] /= torch.sum(grad_hists[n, t, k, c], dim=-1)

    #color_hists = color_hists.view((N, T, K, C * n_bins)) / (Hp * Wp)
    #grad_hists = grad_hists.view((N, T, K, C * n_bins)) / (Hp * Wp)

    color_dists = torch.norm(color_hists.unsqueeze(2) - color_hists.unsqueeze(3),
                             dim=[-1], p=p)   # (N, T, K, K)

    grad_dists = torch.norm(grad_hists.unsqueeze(2) - grad_hists.unsqueeze(3),
                            dim=[-1], p=p)  # (N, T, K, K)

    color_dists = torch.sum(color_dists, dim=[-1, -2])/(K * (K-1))  # (N, T)

    grad_dists = torch.sum(grad_dists, dim=[-1, -2])/(K * (K-1))  # (N, T)

    return torch.mean(color_dists) * torch.mean(grad_dists)

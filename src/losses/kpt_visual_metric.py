import torch

from kornia.filters import sobel
from src.utils.kpt_utils import get_image_patches


def kpt_visual_metric(kpt_sequence: torch.Tensor,
                      img_sequence: torch.Tensor,
                      patch_size: tuple,
                      n_bins: int = 30) -> torch.Tensor:
    """ Evaluates the visual difference between the different objects tracked among key-points.



    :param kpt_sequence:
    :param img_sequence:
    :param patch_size:
    :param n_bins:
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
                    grad_hists[n, t, k, c] = torch.histc(grads[n, t, k, c], bins=n_bins)

    # Get distances
    color_dist = torch.empty(N, K, T - 1)
    grad_dist = torch.empty(N, K, T - 1)

    for k in range(K):
        for t in range(T - 1):
            color_dist[:, k, t] = torch.norm(color_hists[:, t:t + 1, k, :] - color_hists[:, t + 1:t + 2, k, :],
                                             p=float('inf'))
            grad_dist[:, k, t] = torch.norm(grad_hists[:, t:t + 1, k, :] - grad_hists[:, t + 1:t + 2, k, :],
                                            p=float('inf'))

    return color_dist.mean() + grad_dist.mean()

    return ...

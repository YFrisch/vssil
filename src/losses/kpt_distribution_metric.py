import torch
from torch.distributions import MultivariateNormal

from src.utils.kpt_utils import kpts_2_img_coordinates


def kpt_distribution_metric(kpt_sequence: torch.Tensor,
                            img_shape: tuple,
                            n_samples: int) -> torch.Tensor:
    """ Evaluates the consistency of the distribution of key-points
        in the image, across time.

        The metric assumes a multivariate gaussian distribution with means
        at the key-points positions and compares these distributions across
        time using the max-norm difference along a set of samples.

    :param kpt_sequence: Torch tensor of key-point coordinates in (N, T, K, D)
    :param img_shape: Shape of the images (H, W)
    :param n_samples: Number of samples to generate from the MV Gaussian
    :return:
    """

    N, T, K, D = kpt_sequence.shape
    H, W = img_shape

    kpt_sequence = kpts_2_img_coordinates(kpt_sequence, img_shape)

    # Torch tensor holding samples
    kpt_pos_samples = torch.empty((N, T, K, n_samples, 2))

    for t in range(T):

        # Create MV Gaussian from current key-point positions
        # Variance is set to be roughly 10% of image size
        mvN = MultivariateNormal(loc=kpt_sequence[:, t, :, :2],
                                 covariance_matrix=torch.eye(2)*int(0.1*H))

        # Sample from MV Gaussian
        for s in range(n_samples):
            kpt_pos_samples[:, t, :, s, :] = mvN.sample()

    # Calculate max-norm distances over time
    dist = None
    for t in range(T-1):
        d = torch.norm(kpt_pos_samples[:, t:t+1, ...] - kpt_pos_samples[:, t+1:t+2, ...],
                       p=float('inf'),
                       dim=[-3, -2, -1])

        # Normalize by image shape
        d /= W

        dist = d if dist is None else torch.cat([dist, d], dim=1)  # (N, 1)

    # Sum over time
    dist = torch.sum(dist, dim=1)/T  # (N,)

    # Average over batch
    dist = torch.mean(dist, dim=0)  # (1,)

    return dist



"""

    Time-Contrastive (Triplet) Loss for key-point detection.

    For a series of key-points, the coordinates at each frame are used as anchor (a).
    Then, within a temporal range (alpha), the positive match (p) is chosen to be the hardest positive example,
    by calculating argmax ||f(a) - f(p)||.
    Analogously the negative non-match (n) is chosen to be the hardest negative example, yielding the minimum
    to argmin ||f(a) - f(n)||.

"""

import torch
import numpy as np


def time_contrastive_triplet_loss(coords: torch.Tensor, cfg: dict) -> torch.Tensor:
    """ Encourages key-points to have smoother trajectories.

    :param coords: Series of key-point coordinates in (N, T, C, 2/3)
    :param cfg: Configuration dictionary
    :return: The TC triplet loss
    """

    alpha = np.floor(((cfg['training']['tc_loss_alpha'] - 1)/2))
    eps = 0.5
    # eps = alpha

    loss = 0

    N, T = coords.shape[0:2]

    for n in range(N):

        # Iterate over anchors / frames
        for t in range(T):
            f_anchor = coords[n, t, ...]

            pos_indices = np.arange(int(np.clip(a=t-alpha, a_min=0, a_max=T)),
                                    int(np.clip(a=t+alpha+1, a_min=0, a_max=T)))
            pt = torch.argmax(torch.norm(input=coords[n, pos_indices, ...] - f_anchor, p=2)**2, dim=0)
            pt = pos_indices[pt]
            f_positive = coords[n, pt, ...]

            neg_indices = np.concatenate([np.arange(0, pos_indices[0]), np.arange(pos_indices[-1]+1, T)])
            nt = torch.argmin(torch.norm(input=coords[n, neg_indices, ...] - f_anchor, p=2)**2, dim=0)
            nt = neg_indices[nt]
            f_negative = coords[n, nt, ...]

            L_pos = torch.norm(f_anchor - f_positive, p=2)
            L_neg = torch.norm(f_anchor - f_negative, p=2)

            #print(f'+: {L_pos}\t -: {L_neg}')

            loss = loss + torch.max(torch.norm(f_anchor - f_positive, p=2) - torch.norm(f_anchor - f_negative, p=2) + eps,
                                    torch.tensor([0.0]))

    loss = loss / (T * N)

    return torch.tensor([loss]).to(coords.device)


if __name__ == "__main__":

    fake_cfg = {
        'training': {
            'tc_loss_alpha': 9
        }
    }

    # Key-points moving away from each other
    fake_coords = torch.rand(size=(1, 16, 3, 3))
    fake_coords[..., 2] = 1.0
    for t in range(16):
        fake_coords[0, t, 0, :2] += torch.tensor([0.1, 0.1])
        fake_coords[0, t, 1, :2] += torch.tensor([0.1, -0.1])
        fake_coords[0, t, 2, :2] += torch.tensor([-0.1, -0.1])

    print(time_contrastive_triplet_loss(fake_coords, fake_cfg))

    # Key-points staying close
    fake_coords = torch.rand(size=(1, 16, 3, 3))
    fake_coords[..., 2] = 1.0
    for t in range(16):
        fake_coords[0, t, 0, :2] += torch.tensor([0.1, 0.1])
        fake_coords[0, t, 1, :2] += torch.tensor([0.1, 0])
        fake_coords[0, t, 2, :2] += torch.tensor([0.1, -0.1])

    print(time_contrastive_triplet_loss(fake_coords, fake_cfg))

    # Key-points staying even closer
    fake_coords = torch.rand(size=(1, 16, 3, 3))
    fake_coords[..., 2] = 1.0
    for t in range(16):
        fake_coords[0, t, 0, :2] += torch.tensor([0.1, 0.01])
        fake_coords[0, t, 1, :2] += torch.tensor([0.1, 0])
        fake_coords[0, t, 2, :2] += torch.tensor([0.1, -0.01])

    print(time_contrastive_triplet_loss(fake_coords, fake_cfg))

    # Two key-points staying together
    fake_coords = torch.rand(size=(1, 16, 3, 3))
    fake_coords[..., 2] = 1.0
    for t in range(16):
        fake_coords[0, t, 0, :2] += torch.tensor([0.1, 0.01])
        fake_coords[0, t, 1, :2] += torch.tensor([0.1, 0])
        fake_coords[0, t, 2, :2] += torch.tensor([0.1, 0])

    print(time_contrastive_triplet_loss(fake_coords, fake_cfg))

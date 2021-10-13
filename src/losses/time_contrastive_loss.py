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

    alpha = np.floor(((cfg['training']['tc_loss_alpha']-1)/2))
    eps = 0.5
    # eps = alpha

    loss = 0

    N = coords.shape[0]
    T = coords.shape[1]
    for n in range(N):

        # Iterate over anchors / frames
        for t in range(T):
            f_a = coords[n, t, ...]

            pos_indices = np.arange(int(np.clip(a=t-alpha, a_min=0, a_max=T)),
                                    int(np.clip(a=t+alpha+1, a_min=0, a_max=T)))
            pt = torch.argmax(torch.norm(input=coords[n, pos_indices, ...] - f_a, p=2)**2, dim=0)
            pt = pos_indices[pt]
            f_p = coords[n, pt, ...]

            neg_indices = np.concatenate([np.arange(0, pos_indices[0]), np.arange(pos_indices[-1]+1, T)])
            nt = torch.argmin(torch.norm(input=coords[n, neg_indices, ...] - f_a, p=2)**2, dim=0)
            nt = neg_indices[nt]
            f_n = coords[n, nt, ...]

            #print(torch.norm(f_a - f_p, p=2)**2 - torch.norm(f_a - f_n, p=2)**2)

            loss = loss + torch.norm(f_a - f_p, p=2)**2 - torch.norm(f_a - f_n, p=2)**2 + eps

            #print(f't: {t}\t +: {pos_indices}\t -: {neg_indices}\t pt: {pt}\t nt: {nt}')
            #print(f'loss: {loss}')
            #print()

    loss = loss / (T * N)

    return torch.tensor([loss]).to(coords.device)


if __name__ == "__main__":

    fake_cfg = {
        'training': {
            'tc_loss_alpha': 5
        }
    }

    # Abrupt movement
    fake_coords = torch.zeros(size=(16, 15, 3, 3))
    fake_coords[..., 2] = 1
    fake_coords[0, 5:10, :, :2] = 0.25
    fake_coords[0, 10:, :, :2] = 0.75

    # Very abrupt movement
    fake_coords5 = torch.zeros(size=(16, 15, 3, 3))
    fake_coords5[..., 2] = 1
    fake_coords5[0, 10:, :, :2] = 0.9

    # Smooth movement
    fake_coords2 = torch.zeros(size=(16, 15, 3, 3))
    fake_coords2[..., 2] = 1
    fake_coords2[0, 0:2, :, :2] = 0.1
    fake_coords2[0, 2:4, :, :2] = 0.2
    fake_coords2[0, 4:6, :, :2] = 0.3
    fake_coords2[0, 6:8, :, :2] = 0.4
    fake_coords2[0, 8:10, :, :2] = 0.5
    fake_coords2[0, 10:12, :, :2] = 0.6
    fake_coords2[0, 12:-1, :, :2] = 0.5

    # Random coordinates
    fake_coords3 = torch.rand(size=(16, 15, 3, 3))

    # No movement
    fake_coords4 = torch.ones_like(fake_coords3)*0.5

    print('Abrupt movement: ', time_contrastive_triplet_loss(fake_coords, fake_cfg))
    print('Very abrupt movement: ', time_contrastive_triplet_loss(fake_coords5, fake_cfg))
    print('Smooth movement: ', time_contrastive_triplet_loss(fake_coords2, fake_cfg))
    print('Random coordinates: ', time_contrastive_triplet_loss(fake_coords3, fake_cfg))
    print('No movement: ', time_contrastive_triplet_loss(fake_coords4, fake_cfg))

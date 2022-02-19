import torch

import matplotlib.pyplot as plt

from src.utils.kpt_utils import kpts_2_img_coordinates


def kpt_rod_metric(kpt_sequence: torch.Tensor,
                   image_sequence: torch.Tensor,
                   diameter: int,
                   mask_threshold: float = 0.3) -> torch.Tensor:
    """ This method evaluates how precisely the key-points are positioned
        on regions of differences between image frames.

        Assuming a (mostly) static background, the number of pixels around
        a key-point are counted, if they are in a region of difference.

        A 'good' key-point will have a high percentage, closer to 100%,
        while a 'bad' key-point will have lower percentages, closer to 0%.

    :param kpt_sequence: Torch tensor of key-point coordinates in (N, T, K, D)
    :param image_sequence: Torch tensor of images in (N, T, C, H, W)
    :param diameter: Diameter of the region around key-points to consider
                     Should be roughly 1/20 of image size in most cases.
    :param mask_threshold:
    :return:
    """

    N, T, C, H, W = image_sequence.shape
    _, _, K, D = kpt_sequence.shape

    img_coordinate_sequence = kpts_2_img_coordinates(kpt_sequence, (H, W))

    pixel_count = torch.zeros((N, T-1, K, 2))

    for n in range(N):
        for t in range(0, T - 1):
            d = image_sequence[n, t] - image_sequence[n, t + 1]
            # Filter frame differences to distinguish between background and foreground
            d_mask = torch.where((torch.abs(torch.norm(d, dim=0, p=2)) > mask_threshold), 1.0, 0.0)
            #plt.imshow(d_mask, cmap='gray')

            for k in range(K):
                kpt_w_1, kpt_h_1 = img_coordinate_sequence[n, t, k, 0], img_coordinate_sequence[n, t, k, 1]
                w_min_1, w_max_1 = max(0, int(kpt_w_1 - 0.5 * diameter)), min(W - 1, int(kpt_w_1 + 0.5 * diameter))
                h_min_1, h_max_1 = max(0, int(kpt_h_1 - 0.5 * diameter)), min(H - 1, int(kpt_h_1 + 0.5 * diameter))
                pixel_count[n, t, k, 0] += torch.sum(d_mask[h_min_1:h_max_1, w_min_1:w_max_1])

                kpt_w_2, kpt_h_2 = img_coordinate_sequence[0, t + 1, k, 0], img_coordinate_sequence[0, t + 1, k, 1]
                w_min_2, w_max_2 = max(0, int(kpt_w_2 - 0.5 * diameter)), min(W - 1, int(kpt_w_2 + 0.5 * diameter))
                h_min_2, h_max_2 = max(0, int(kpt_h_2 - 0.5 * diameter)), min(H - 1, int(kpt_h_2 + 0.5 * diameter))
                pixel_count[n, t, k, 1] += torch.sum(d_mask[h_min_2:h_max_2, w_min_2:w_max_2])

                #plt.scatter(img_coordinate_sequence[n, t, k, 0], img_coordinate_sequence[n, t, k, 1])
                #plt.scatter(img_coordinate_sequence[n, t+1, k, 0], img_coordinate_sequence[n, t+1, k, 1])
            #plt.show()
            #exit()
    #print(pixel_count.shape)
    #print(pixel_count)
    #exit()

    # Normalize count by diameter
    normalized_pixel_count = pixel_count / (diameter * diameter)

    # Normalize (twice) by number of key-points
    normalized_pixel_count = torch.sum(normalized_pixel_count, dim=[-2, -1])/(2 * K)

    # Average across batch and time-steps
    return torch.mean(normalized_pixel_count)

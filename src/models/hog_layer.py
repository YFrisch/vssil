import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import matplotlib.pyplot as plt
from torchvision import transforms as t
from PIL import Image


class HoGLayer(nn.Module):

    """ Implementation of a differentiable HoG layer, following
        https://towardsdatascience.com/hog-histogram-of-oriented-gradients-67ecd887675f
    """
    
    def __init__(self,
                 img_shape: tuple,
                 cell_size: tuple = (8, 8),
                 n_bins: int = 9):

        super(HoGLayer, self).__init__()

        # Kernels for gradient calculations
        vertical_kernel = [[0, -1, 0], [0, 0, 0], [0, 1, 0]]
        horizontal_kernel = [[0, 0, 0], [-1, 0, 1], [0, 0, 0]]
        vertical_kernel = torch.FloatTensor(vertical_kernel)[None, None, ...].repeat(1, 3, 1, 1)
        horizontal_kernel = torch.FloatTensor(horizontal_kernel)[None, None, ...].repeat(1, 3, 1, 1)
        self.weight_vertical = nn.Parameter(data=vertical_kernel, requires_grad=False)
        self.weight_horizontal = nn.Parameter(data=horizontal_kernel, requires_grad=False)

        """
        self.conv2d = nn.Conv2d(
            3, 3, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2d_T = nn.ConvTranspose2d(
            3, 3, kernel_size=3, stride=1, padding=1, bias=False
        )
        """

        self.img_shape = img_shape
        self.cell_size = cell_size
        self.n_bins = n_bins
        self.delta = 180 / self.n_bins
        self.epsilon = 1e-9

    def jth_bin(self, angle: float) -> float:
        temp = (angle / self.delta) - 0.5
        return math.floor(temp)

    def jth_bin_centre(self, j: int) -> float:
        C_j = self.delta*(j + 0.5)
        return round(C_j, 9)

    def jth_bin_value(self, mag: float, ang: float, j: int) -> float:
        C_j = self.jth_bin_centre(j + 1)
        V_j = mag * ((C_j - ang) / self.delta)
        return round(V_j, 9)

    def hists(self, magnitude: torch.Tensor, angle: torch.Tensor) -> list:
        """ Calculates the angle/magnitude histograms per cell.

        :param magnitude: Torch tensor of magnitudes per pixel in (H, W)
        :param angle: Torch tensor of angles per pixel in (H, W)
        :return: [[[vote for each bin] for each column] for each row]
        """
        histograms = []

        for i in range(0, self.img_shape[0], self.cell_size[0]):
            temp = []
            for j in range(0, self.img_shape[1], self.cell_size[1]):
                magnitude_values = [[magnitude[v][h].item() for h in range(j, j + self.cell_size[1])]
                                    for v in range(i, i + self.cell_size[0])]
                angle_values = [[angle[v][h].item() for h in range(j, j + self.cell_size[1])]
                                for v in range(i, i + self.cell_size[0])]
                for k in range(len(magnitude_values)):
                    for l in range(len(magnitude_values[0])):
                        # Init empty histogram
                        bins = [0.0 for _ in range(self.n_bins)]
                        value_j = self.jth_bin(angle_values[k][l])
                        Vj = self.jth_bin_value(magnitude_values[k][l], angle_values[k][l], value_j)
                        Vj_1 = magnitude_values[k][l] - Vj
                        bins[value_j] += Vj
                        bins[value_j + 1] += Vj_1
                        #bins = [round(x, 9) for x in bins]
                temp.append(bins)
            histograms.append(temp)

        return histograms

    def features(self, histogram_points_nine: list) -> torch.Tensor:
        """ Calculates features after block-normalization.

        :param histogram_points_nine:
        :return:
        """

        feature_vectors = []

        for i in range(0, len(histogram_points_nine) - 1, 1):
            temp = []
            for j in range(0, len(histogram_points_nine[0]) - 1, 1):
                values = [[histogram_points_nine[i][x] for x in range(j, j + 2)] for i in range(i, i + 2)]
                final_vector = []
                for k in values:
                    for l in k:
                        for m in l:
                            final_vector.append(m)
                hist_norm = round(math.sqrt(sum([pow(x, 2) for x in final_vector])), 9)
                final_vector = [round(x / (hist_norm + self.epsilon), 9) for x in final_vector]
                temp.append(final_vector)
            feature_vectors.append(temp)

        return torch.tensor(feature_vectors)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        assert x.dim() == 4
        assert x.shape[1] in [1, 3]

        assert x.shape[2] % self.cell_size[0] == 0, "Chose proper cell size"
        assert x.shape[3] % self.cell_size[1] == 0, "Chose proper cell size"

        n_rows = int(x.shape[2]/self.cell_size[0]) - 1
        n_columns = int(x.shape[3]/self.cell_size[1]) - 1

        # Vertical gradients
        G_v = f.conv2d(x, weight=self.weight_vertical, bias=None, stride=1, padding=1)

        # Horizontal gradients
        G_h = f.conv2d(x, weight=self.weight_horizontal, bias=None, stride=1, padding=1)

        # Magnitude per pixel
        mag = torch.sqrt(torch.pow(G_v, 2) + torch.pow(G_h, 2) + 1e-6)

        # Angle per pixel
        ang = torch.abs(torch.arctan(G_h / (G_v + self.epsilon))*(180.0/np.pi))

        """
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(x.squeeze(0).permute(1, 2, 0).detach().cpu().numpy())
        ax[1].imshow(mag.squeeze(0).permute(1, 2, 0).detach().cpu().numpy(), cmap='gray')
        ax[2].imshow(ang.squeeze(0).permute(1, 2, 0).detach().cpu().numpy(), cmap='gray')
        plt.show()
        """

        hists = self.hists(magnitude=mag.squeeze(), angle=ang.squeeze())

        features = self.features(histogram_points_nine=hists)

        return features, n_rows, n_columns


if __name__ == "__main__":

    # Note: The patch shape needs to have a 2:1 ratio!
    #test_img_shape = (256, 128)
    test_img_shape = (128, 64)
    test_img = Image.open('tests/runner.png')
    test_img = t.PILToTensor()(test_img)

    if test_img.shape[0] == 0:
        test_img.unsqueeze(0)
    else:
        test_img = test_img[:-1, ...]

    test_img = t.Resize(size=test_img_shape)(test_img).float() / 255.0
    #test_img_rotated = t.RandomHorizontalFlip(p=1.0)(test_img)
    test_img_rotated = t.RandomVerticalFlip(p=1.0)(test_img)
    #test_img_rotated = t.RandomPerspective(p=1.0, fill=1)(test_img)

    test_img2 = Image.open('tests/micky.png')
    test_img2 = t.PILToTensor()(test_img2)

    if test_img2.shape[0] == 0:
        test_img2.unsqueeze(0)
    else:
        test_img2 = test_img2[:-1, ...]

    test_img2 = t.Resize(size=test_img_shape)(test_img2).float() / 255.0
    test_img2_rotated = t.RandomVerticalFlip(p=1.0)(test_img2)
    #test_img2_rotated = t.RandomPerspective(p=1.0)(test_img2)

    random_noise = torch.rand_like(test_img_rotated)

    hog = HoGLayer(img_shape=test_img_shape)

    out, n_rows, n_cols = hog(test_img.unsqueeze(0))

    out_rot, _, _ = hog(test_img_rotated.unsqueeze(0))
    out2, _, _ = hog(test_img2.unsqueeze(0))
    out2_rot, _, _ = hog(test_img2_rotated.unsqueeze(0))
    out_rand, _, _ = hog(random_noise.unsqueeze(0))

    print('||f(1) - f(1`)||: ', torch.abs(torch.sum(out) - torch.sum(out_rot)))
    print('||f(2) - f(2`)||: ', torch.abs(torch.sum(out2) - torch.sum(out2_rot)))
    print('||f(1) - f(2)||: ', torch.abs(torch.sum(out) - torch.sum(out2)))
    print('||f(1) - f(r)||: ', torch.abs(torch.sum(out) - torch.sum(out_rand)))
    print('||f(2) - f(r)||: ', torch.abs(torch.sum(out2) - torch.sum(out_rand)))

import time
import unittest
import random

import torch
from torch.autograd import gradcheck
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from src.losses.pixelwise_contrastive_loss import pixelwise_contrastive_loss, get_image_patch


def play_img_and_keypoints(image_series: torch.Tensor, kpts: torch.Tensor, title: str):
    fig = plt.figure()
    ax = plt.axes()
    ax.set_title(title)

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=4, metadata=dict(artist='Me'), bitrate=1800)

    N, T, C, H, W = image_series.shape

    if C == 1:
        im_buff = ax.imshow(image_series[0, 0, ...].permute(1, 2, 0).detach().cpu().numpy(), cmap='gray', vmin=0,
                            vmax=1)
    else:
        im_buff = ax.imshow(image_series[0, 0, ...].permute(1, 2, 0).detach().cpu().numpy(), vmin=0, vmax=1)

    scatter_buff = [ax.scatter((3 / 20) * W, (3 / 20) * H) for _ in range(kpts.shape[2])]

    def anim(i):
        im_buff.set_data(image_series[0, i, ...].detach().permute(1, 2, 0).cpu().numpy())
        for k in range(kpts.shape[2]):
            h_k = H * (kpts[0, i, k, 0] + 1) / 2
            w_k = W * (-kpts[0, i, k, 1] + 1) / 2
            scatter_buff[k].set_offsets([w_k, h_k])

    ani = animation.FuncAnimation(fig, anim, frames=20, repeat=False)
    ani.save(f'case_{title}.mp4', writer=writer)

    plt.show()
    plt.close()


class PatchLossTest(unittest.TestCase):

    def setUp(self) -> None:

        torch.manual_seed(123)
        random.seed(123)
        np.random.seed(123)

        # Fake image series with 3 moving squares
        # fake_img_series = torch.zeros(size=(1, 20, 3, 32, 32), dtype=torch.double)
        fake_img_series = torch.zeros(size=(1, 20, 3, 32, 32))
        for t in range(0, fake_img_series.shape[1]):
            fake_img_series[:, t, :, t + 1:t + 5, t + 1:t + 5] = 0.5
            fake_img_series[:, t, :, t + 1:t + 5, 1:5] = 0.2
            fake_img_series[:, t, :, 1:5, t + 1:t + 5] = 0.8

        self.fake_img_series = fake_img_series
        #self.fake_img_series.requires_grad_(True)

    def each_kp_diff_patch(self, image_series: torch.Tensor) -> torch.Tensor:
        # All key-points on a different patch
        #kpts = torch.zeros(size=(1, 20, 3, 3), dtype=torch.double)
        kpts = torch.zeros(size=(1, 20, 3, 3))
        kpts[..., 2] = 1
        for t in range(0, image_series.shape[1]):
            kpts[0, t, 0, 0] = -0.85 + (1 / 16 * t)
            kpts[0, t, 0, 1] = 0.85 - (1 / 16 * t)
            kpts[0, t, 1, 0] = -0.85 + (1 / 16 * t)
            kpts[0, t, 1, 1] = 0.85
            kpts[0, t, 2, 0] = - 0.85
            kpts[0, t, 2, 1] = 0.85 - (1 / 16 * t)
        kpts = kpts[..., :2] + torch.rand_like(kpts[..., :2]) * 0.05
        return kpts

    def two_kpts_same_patch(self, image_series: torch.Tensor) -> torch.Tensor:
        # Two key-points on the same patch
        kpts = torch.zeros(size=(1, 20, 3, 3))
        kpts[..., 2] = 1
        for t in range(0, image_series.shape[1]):
            kpts[0, t, 0, 0] = -0.85 + (1 / 16 * t)
            kpts[0, t, 0, 1] = 0.85 - (1 / 16 * t)
            kpts[0, t, 1:3, 0] = -0.85 + (1 / 16 * t)
            kpts[0, t, 1:3, 1] = 0.85
        kpts = kpts[..., :2] + torch.rand_like(kpts[..., :2]) * 0.05
        return kpts

    def all_kpts_same_patch(self, image_series: torch.Tensor) -> torch.Tensor:
        # All key-points on the same patch
        kpts = torch.zeros(size=(1, 20, 3, 3))
        kpts[..., 2] = 1
        for t in range(0, image_series.shape[1]):
            kpts[0, t, :, 0] = -0.85 + (1 / 16 * t)
            kpts[0, t, :, 1] = 0.85 - (1 / 16 * t)
        kpts = kpts[..., :2] + torch.rand_like(kpts[..., :2]) * 0.05
        return kpts

    def kpts_changing_patches(self, image_series) -> torch.Tensor:
        # Key-points changing patches mid sequence
        kpts = torch.zeros(size=(1, 20, 3, 3))
        kpts[..., 2] = 1
        for t in range(0, image_series.shape[1]):
            if t < kpts.shape[1] / 2:
                kpts[0, t, 0, 0] = -0.85 + (1 / 16 * t)
                kpts[0, t, 0, 1] = 0.85 - (1 / 16 * t)
                kpts[0, t, 1, 0] = -0.85 + (1 / 16 * t)
                kpts[0, t, 1, 1] = 0.85
                kpts[0, t, 2, 0] = - 0.85
                kpts[0, t, 2, 1] = 0.85 - (1 / 16 * t)
            else:
                kpts[0, t, 0, 0] = -0.85 + (1 / 16 * t)
                kpts[0, t, 0, 1] = 0.85 - (1 / 16 * t)
                kpts[0, t, 1, 0] = -0.85
                kpts[0, t, 1, 1] = 0.85 - (1 / 16 * t)
                kpts[0, t, 2, 0] = - 0.85 + (1 / 16 * t)
                kpts[0, t, 2, 1] = 0.85
        kpts = kpts[..., :2] + torch.rand_like(kpts[..., :2]) * 0.05
        return kpts

    def kpts_often_changing_patches(self, image_series):
        # Key-points changing patches mid sequence
        kpts = torch.zeros(size=(1, 20, 3, 3))
        kpts[..., 2] = 1
        for t in range(0, image_series.shape[1]):
            if t < kpts.shape[1] / 3:
                kpts[0, t, 0, 0] = -0.85 + (1 / 16 * t)
                kpts[0, t, 0, 1] = 0.85 - (1 / 16 * t)
                kpts[0, t, 1, 0] = -0.85 + (1 / 16 * t)
                kpts[0, t, 1, 1] = 0.85
                kpts[0, t, 2, 0] = - 0.85
                kpts[0, t, 2, 1] = 0.85 - (1 / 16 * t)
            elif kpts.shape[1] / 3 <= t < 2 * kpts.shape[1] / 3:
                kpts[0, t, 0, 0] = -0.85 + (1 / 16 * t)
                kpts[0, t, 0, 1] = 0.85 - (1 / 16 * t)
                kpts[0, t, 1, 0] = -0.85
                kpts[0, t, 1, 1] = 0.85 - (1 / 16 * t)
                kpts[0, t, 2, 0] = - 0.85 + (1 / 16 * t)
                kpts[0, t, 2, 1] = 0.85
            else:
                kpts[0, t, 0, 0] = - 0.85 + (1 / 16 * t)
                kpts[0, t, 0, 1] = 0.85
                kpts[0, t, 1, 0] = -0.85
                kpts[0, t, 1, 1] = 0.85 - (1 / 16 * t)
                kpts[0, t, 2, 0] = -0.85 + (1 / 16 * t)
                kpts[0, t, 2, 1] = 0.85 - (1 / 16 * t)
        kpts = kpts[..., :2] + torch.rand_like(kpts[..., :2]) * 0.05
        return kpts

    def testContrast(self):
        """

            The key-points should represent different objects in a scene.


        """

        fake_fnn = torch.nn.Linear(in_features=20*3*2, out_features=20*3*2, bias=False)
        fake_fnn2 = torch.nn.Linear(in_features=20*3*2, out_features=20*3*2, bias=False)

        patch_size = (5, 5)
        time_window = 9
        alpha = 0.5

        fake_kpts = self.each_kp_diff_patch(self.fake_img_series)
        print(fake_kpts.shape)
        print(torch.flatten(fake_kpts, start_dim=1).shape)
        fake_kpts = fake_fnn2(fake_fnn(torch.flatten(fake_kpts, start_dim=1))).view(1, 20, 3, 2)
        #fake_kpts.requires_grad_(True)
        fake_kpts2 = self.two_kpts_same_patch(self.fake_img_series)
        #fake_kpts2.requires_grad_(True)
        fake_kpts3 = self.all_kpts_same_patch(self.fake_img_series)
        #fake_kpts3.requires_grad_(True)

        # play_img_and_keypoints(self.fake_img_series, fake_kpts, title='1')
        # play_img_and_keypoints(self.fake_img_series, fake_kpts2, title='2')
        # play_img_and_keypoints(self.fake_img_series, fake_kpts3, title='3')

        """
        print(gradcheck(lambda kpts, img: pixelwise_contrastive_loss(kpts, img, patch_size, time_window, alpha),
                        inputs=(fake_kpts, self.fake_img_series),
                        ))
        exit()
        """

        print('\n##### All key-points on different objects')
        L1 = pixelwise_contrastive_loss(keypoint_coordinates=fake_kpts,
                                        image_sequence=self.fake_img_series,
                                        patch_size=patch_size,
                                        time_window=time_window,
                                        alpha=alpha,
                                        patch_diff_mode='TFeat')
        print(L1)
        L1.retain_grad()
        L1.backward()
        print(L1.grad)
        fake_fnn.weight.retain_grad()
        print(fake_fnn.weight.grad)
        fake_fnn2.weight.retain_grad()
        print(fake_fnn.weight.grad)
        exit()

        print('##### Two key-points on the same object')
        L2 = pixelwise_contrastive_loss(keypoint_coordinates=fake_kpts2,
                                        image_sequence=self.fake_img_series,
                                        patch_size=patch_size,
                                        time_window=time_window,
                                        alpha=alpha,
                                        patch_diff_mode='TFeat')
        print(L2)

        print('##### All key-points on the same object')
        L3 = pixelwise_contrastive_loss(keypoint_coordinates=fake_kpts3,
                                        image_sequence=self.fake_img_series,
                                        patch_size=patch_size,
                                        time_window=time_window,
                                        alpha=alpha,
                                        patch_diff_mode='TFeat')
        print(L3)
        print()

        time.sleep(0.1)

        assert L1 <= L2 <= L3

    def testConsistency(self):
        """

            The key-points of a scene should consistently represent the same objects over time.


        """

        patch_size = (5, 5)
        time_window = 9
        alpha = 0.5

        fake_kpts = self.each_kp_diff_patch(self.fake_img_series)
        fake_kpts4 = self.kpts_changing_patches(self.fake_img_series)
        fake_kpts5 = self.kpts_often_changing_patches(self.fake_img_series)

        # play_img_and_keypoints(self.fake_img_series, fake_kpts, title='1')
        # play_img_and_keypoints(self.fake_img_series, fake_kpts4, title='4')
        # play_img_and_keypoints(self.fake_img_series, fake_kpts5, title='5')

        print('\n##### Key-points consistently on same object')
        L1 = pixelwise_contrastive_loss(keypoint_coordinates=fake_kpts,
                                        image_sequence=self.fake_img_series,
                                        patch_size=patch_size,
                                        time_window=time_window,
                                        alpha=alpha,
                                        patch_diff_mode='TFeat')
        print(L1)

        print('##### Key-points changing objects mid series')
        L4 = pixelwise_contrastive_loss(keypoint_coordinates=fake_kpts4,
                                        image_sequence=self.fake_img_series,
                                        patch_size=patch_size,
                                        time_window=time_window,
                                        alpha=alpha,
                                        patch_diff_mode='TFeat')
        print(L4)

        print('##### Key-points changing objects multiple times')
        L5 = pixelwise_contrastive_loss(keypoint_coordinates=fake_kpts5,
                                        image_sequence=self.fake_img_series,
                                        patch_size=patch_size,
                                        time_window=time_window,
                                        alpha=alpha,
                                        patch_diff_mode='TFeat')
        print(L5)

        time.sleep(0.1)
        print()

        assert L1 <= L4 <= L5

    def testGradientFlow(self):

        fake_fnn = torch.nn.Linear(in_features=20 * 3 * 2, out_features=20 * 3 * 2, bias=False)
        fake_fnn2 = torch.nn.Linear(in_features=20 * 3 * 2, out_features=20 * 3 * 2, bias=False)

        patch_size = (5, 5)
        time_window = 9
        alpha = 0.5

        self.fake_img_series.requires_grad_(True)
        fake_kpts = self.each_kp_diff_patch(self.fake_img_series)
        fake_kpts.requires_grad_(True)
        fake_kpts = fake_fnn2(fake_fnn(torch.flatten(fake_kpts, start_dim=1))).view(1, 20, 3, 2)
        fake_patch = get_image_patch(keypoint_coordinates=fake_kpts[0, 10:11, 0, :],
                                     image=self.fake_img_series[0, 10:11, ...],
                                     patch_size=(5, 5))
        L = fake_patch.norm(p=2)
        grad = L.backward()
        print(grad)


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np
import torch
from torchvision.io import read_video

from src.losses.kpt_metrics import spatial_consistency_loss, combined_metric, patchwise_contrastive_metric
from src.losses.pixelwise_contrastive_loss_2 import pixelwise_contrastive_loss


class Keypoint_Metric_Test(unittest.TestCase):

    def setUp(self) -> None:

        self.freq = 10

        self.img_sequence_perfect_kpts = None
        self.perfect_kpts_1 = None
        self.perfect_kpts_2 = None
        self.perfect_kpts_3 = None

        self.img_sequence_noisy_kpts = None
        self.noisy_kpts_1 = None
        self.noisy_kpts_2 = None
        self.noisy_kpts_3 = None

        self.img_sequence_laggy_kpts = None
        self.laggy_kpts_1 = None
        self.laggy_kpts_2 = None
        self.laggy_kpts_3 = None

        self.img_sequence_changing_kpts = None
        self.changing_kpts_1 = None
        self.changing_kpts_2 = None
        self.changing_kpts_3 = None

    def read_perfect(self):

        print("##### Read-in perfect key-points data.")

        self.img_sequence_perfect_kpts = read_video(
            '/home/yannik/vssil/test_data/smooth.mp4',
            pts_unit='sec'
        )[0].permute((0, 3, 1, 2))[::self.freq, ...].unsqueeze(0).float()/255.0  # (N, T, C, H, W)

        with open('/home/yannik/vssil/test_data/smooth_0.txt') as perfect_kpts1_file:

            for line in perfect_kpts1_file.readlines():
                l = line.replace("\t", "").replace("\n", "").split(";")
                self.perfect_kpts_1 = torch.tensor([float(l[0]), float(l[1]), float(l[2])]).view((1, 1, 1, 3)) \
                    if self.perfect_kpts_1 is None else torch.cat([
                    self.perfect_kpts_1, torch.tensor([float(l[0]), float(l[1]), float(l[2])]).view((1, 1, 1, 3))
                ], dim=1)

        self.perfect_kpts_1 = self.convert_kpts(self.perfect_kpts_1[:, ::self.freq, ...],
                                                self.img_sequence_perfect_kpts.shape[3],
                                                self.img_sequence_perfect_kpts.shape[4])

        with open('/home/yannik/vssil/test_data/smooth_1.txt') as perfect_kpts2_file:

            for line in perfect_kpts2_file.readlines():
                l = line.replace("\t", "").replace("\n", "").split(";")
                self.perfect_kpts_2 = torch.tensor([float(l[0]), float(l[1]), float(l[2])]).view((1, 1, 1, 3)) \
                    if self.perfect_kpts_2 is None else torch.cat([
                    self.perfect_kpts_2, torch.tensor([float(l[0]), float(l[1]), float(l[2])]).view((1, 1, 1, 3))
                ], dim=1)
        self.perfect_kpts_2 = self.convert_kpts(self.perfect_kpts_2[:, ::self.freq, ...],
                                                self.img_sequence_perfect_kpts.shape[3],
                                                self.img_sequence_perfect_kpts.shape[4])

        with open('/home/yannik/vssil/test_data/smooth_2.txt') as perfect_kpts3_file:

            for line in perfect_kpts3_file.readlines():
                l = line.replace("\t", "").replace("\n", "").split(";")
                self.perfect_kpts_3 = torch.tensor([float(l[0]), float(l[1]), float(l[2])]).view((1, 1, 1, 3)) \
                    if self.perfect_kpts_3 is None else torch.cat([
                    self.perfect_kpts_3, torch.tensor([float(l[0]), float(l[1]), float(l[2])]).view((1, 1, 1, 3))
                ], dim=1)
        self.perfect_kpts_3 = self.convert_kpts(self.perfect_kpts_3[:, ::self.freq, ...],
                                                self.img_sequence_perfect_kpts.shape[3],
                                                self.img_sequence_perfect_kpts.shape[4])

    def read_laggy(self):

        print("##### Read-in laggy key-points data.")

        self.img_sequence_laggy_kpts = read_video(
            '/home/yannik/vssil/test_data/laggy_kpts.mp4',
            pts_unit='sec'
        )[0].permute((0, 3, 1, 2))[::self.freq, ...].unsqueeze(0).float()/255.0  # (T, C, H, W)

        with open('/home/yannik/vssil/test_data/laggy_kpts_0.txt') as laggy_kpts1_file:

            for line in laggy_kpts1_file.readlines():
                l = line.replace("\t", "").replace("\n", "").split(";")
                self.laggy_kpts_1 = torch.tensor([float(l[0]), float(l[1]), float(l[2])]).view((1, 1, 1, 3)) \
                    if self.laggy_kpts_1 is None else torch.cat([
                    self.laggy_kpts_1, torch.tensor([float(l[0]), float(l[1]), float(l[2])]).view((1, 1, 1, 3))
                ], dim=1)
        self.laggy_kpts_1 = self.convert_kpts(self.laggy_kpts_1[:, ::self.freq, ...],
                                              self.img_sequence_laggy_kpts.shape[3],
                                              self.img_sequence_laggy_kpts.shape[4])

        with open('/home/yannik/vssil/test_data/laggy_kpts_1.txt') as laggy_kpts2_file:

            for line in laggy_kpts2_file.readlines():
                l = line.replace("\t", "").replace("\n", "").split(";")
                self.laggy_kpts_2 = torch.tensor([float(l[0]), float(l[1]), float(l[2])]).view((1, 1, 1, 3)) \
                    if self.laggy_kpts_2 is None else torch.cat([
                    self.laggy_kpts_2, torch.tensor([float(l[0]), float(l[1]), float(l[2])]).view((1, 1, 1, 3))
                ], dim=1)
        self.laggy_kpts_2 = self.convert_kpts(self.laggy_kpts_2[:, ::self.freq, ...],
                                              self.img_sequence_laggy_kpts.shape[3],
                                              self.img_sequence_laggy_kpts.shape[4])

        with open('/home/yannik/vssil/test_data/laggy_kpts_2.txt') as laggy_kpts3_file:

            for line in laggy_kpts3_file.readlines():
                l = line.replace("\t", "").replace("\n", "").split(";")
                self.laggy_kpts_3 = torch.tensor([float(l[0]), float(l[1]), float(l[2])]).view((1, 1, 1, 3)) \
                    if self.laggy_kpts_3 is None else torch.cat([
                    self.laggy_kpts_3, torch.tensor([float(l[0]), float(l[1]), float(l[2])]).view((1, 1, 1, 3))
                ], dim=1)
        self.laggy_kpts_3 = self.convert_kpts(self.laggy_kpts_3[:, ::self.freq, ...],
                                              self.img_sequence_laggy_kpts.shape[3],
                                              self.img_sequence_laggy_kpts.shape[4])

    def read_changing(self):

        print("##### Read-in changing key-points data.")

        self.img_sequence_changing_kpts = read_video(
            '/home/yannik/vssil/test_data/smooth_changing_noisy.mp4',
            pts_unit='sec'
        )[0].permute((0, 3, 1, 2))[::self.freq, ...].unsqueeze(0).float()/255.0  # (T, C, H, W)

        with open('/home/yannik/vssil/test_data/smooth_changing_noisy_0.txt') as changing_kpts1_file:

            for line in changing_kpts1_file.readlines():
                l = line.replace("\t", "").replace("\n", "").split(";")
                self.changing_kpts_1 = torch.tensor([float(l[0]), float(l[1]), float(l[2])]).view((1, 1, 1, 3)) \
                    if self.changing_kpts_1 is None else torch.cat([
                    self.changing_kpts_1, torch.tensor([float(l[0]), float(l[1]), float(l[2])]).view((1, 1, 1, 3))
                ], dim=1)
        self.changing_kpts_1 = self.convert_kpts(self.changing_kpts_1[:, ::self.freq, ...],
                                                 self.img_sequence_changing_kpts.shape[3],
                                                 self.img_sequence_changing_kpts.shape[4])

        with open('/home/yannik/vssil/test_data/smooth_changing_noisy_1.txt') as changing_kpts2_file:

            for line in changing_kpts2_file.readlines():
                l = line.replace("\t", "").replace("\n", "").split(";")
                self.changing_kpts_2 = torch.tensor([float(l[0]), float(l[1]), float(l[2])]).view((1, 1, 1, 3)) \
                    if self.changing_kpts_2 is None else torch.cat([
                    self.changing_kpts_2, torch.tensor([float(l[0]), float(l[1]), float(l[2])]).view((1, 1, 1, 3))
                ], dim=1)
        self.changing_kpts_2 = self.convert_kpts(self.changing_kpts_2[:, ::self.freq, ...],
                                                 self.img_sequence_changing_kpts.shape[3],
                                                 self.img_sequence_changing_kpts.shape[4])

        with open('/home/yannik/vssil/test_data/smooth_changing_noisy_2.txt') as changing_kpts3_file:

            for line in changing_kpts3_file.readlines():
                l = line.replace("\t", "").replace("\n", "").split(";")
                self.changing_kpts_3 = torch.tensor([float(l[0]), float(l[1]), float(l[2])]).view((1, 1, 1, 3)) \
                    if self.changing_kpts_3 is None else torch.cat([
                    self.changing_kpts_3, torch.tensor([float(l[0]), float(l[1]), float(l[2])]).view((1, 1, 1, 3))
                ], dim=1)
        self.changing_kpts_3 = self.convert_kpts(self.changing_kpts_3[:, ::self.freq, ...],
                                                 self.img_sequence_changing_kpts.shape[3],
                                                 self.img_sequence_changing_kpts.shape[4])

    def read_noisy(self):

        print("##### Read-in noisy key-points data.")

        self.img_sequence_noisy_kpts = read_video(
            '/home/yannik/vssil/test_data/smooth_noisy.mp4',
            pts_unit='sec'
        )[0].permute((0, 3, 1, 2))[::self.freq, ...].unsqueeze(0).float()/255.0  # (T, C, H, W)

        with open('/home/yannik/vssil/test_data/smooth_noisy_0.txt') as noisy_kpts1_file:

            for line in noisy_kpts1_file.readlines():
                l = line.replace("\t", "").replace("\n", "").split(";")
                self.noisy_kpts_1 = torch.tensor([float(l[0]), float(l[1]), float(l[2])]).view((1, 1, 1, 3)) \
                    if self.noisy_kpts_1 is None else torch.cat([
                    self.noisy_kpts_1, torch.tensor([float(l[0]), float(l[1]), float(l[2])]).view((1, 1, 1, 3))
                ], dim=1)
        self.noisy_kpts_1 = self.convert_kpts(self.noisy_kpts_1[:, ::self.freq, ...],
                                              self.img_sequence_noisy_kpts.shape[3],
                                              self.img_sequence_noisy_kpts.shape[4])

        with open('/home/yannik/vssil/test_data/smooth_noisy_1.txt') as noisy_kpts2_file:

            for line in noisy_kpts2_file.readlines():
                l = line.replace("\t", "").replace("\n", "").split(";")
                self.noisy_kpts_2 = torch.tensor([float(l[0]), float(l[1]), float(l[2])]).view((1, 1, 1, 3)) \
                    if self.noisy_kpts_2 is None else torch.cat([
                    self.noisy_kpts_2, torch.tensor([float(l[0]), float(l[1]), float(l[2])]).view((1, 1, 1, 3))
                ], dim=1)
        self.noisy_kpts_2 = self.convert_kpts(self.noisy_kpts_2[:, ::self.freq, ...],
                                              self.img_sequence_noisy_kpts.shape[3],
                                              self.img_sequence_noisy_kpts.shape[4])

        with open('/home/yannik/vssil/test_data/smooth_noisy_2.txt') as noisy_kpts3_file:

            for line in noisy_kpts3_file.readlines():
                l = line.replace("\t", "").replace("\n", "").split(";")
                self.noisy_kpts_3 = torch.tensor([float(l[0]), float(l[1]), float(l[2])]).view((1, 1, 1, 3)) \
                    if self.noisy_kpts_3 is None else torch.cat([
                    self.noisy_kpts_3, torch.tensor([float(l[0]), float(l[1]), float(l[2])]).view((1, 1, 1, 3))
                ], dim=1)
        self.noisy_kpts_3 = self.convert_kpts(self.noisy_kpts_3[:, ::self.freq, ...],
                                              self.img_sequence_noisy_kpts.shape[3],
                                              self.img_sequence_noisy_kpts.shape[4])

    def convert_kpts(self, kpts: torch.Tensor, H, W):
        """ Convert pixel coordinates to (-1, +1) coordinates."""
        kpts[..., :2] = kpts[..., :2] / 12.0
        # kpts[..., 0] = (kpts[..., 0] / W) * 2 - 1.0
        # kpts[..., 1] = - (kpts[..., 1] / H) * 2 - 1.0
        return kpts

    def test_perfect_kpts(self):
        self.read_perfect()
        assert self.perfect_kpts_1.shape == self.perfect_kpts_2.shape == self.perfect_kpts_3.shape

    def test_noisy_kpts(self):
        self.read_noisy()
        assert self.noisy_kpts_1.shape == self.noisy_kpts_2.shape == self.noisy_kpts_3.shape

    def test_laggy_kpts(self):
        self.read_laggy()
        assert self.laggy_kpts_1.shape == self.laggy_kpts_2.shape == self.laggy_kpts_3.shape

    def test_changing_kpts(self):
        self.read_changing()
        assert self.changing_kpts_1.shape == self.changing_kpts_2.shape == self.changing_kpts_3.shape

    def test_spatial_consistency_metric(self):
        self.read_perfect()
        self.read_noisy()
        # self.read_laggy()
        # self.read_changing()
        perfect_kpts = torch.cat([self.perfect_kpts_1, self.perfect_kpts_2, self.perfect_kpts_3], dim=2)
        noisy_kpts = torch.cat([self.noisy_kpts_1, self.noisy_kpts_2, self.noisy_kpts_3], dim=2)
        # laggy_kpts = torch.cat([self.laggy_kpts_1, self.laggy_kpts_2, self.laggy_kpts_3], dim=2)
        # changing_kpts = torch.cat([self.changing_kpts_1, self.changing_kpts_2, self.changing_kpts_3], dim=2)
        L_perfect = spatial_consistency_loss(keypoint_coordinates=perfect_kpts)
        print(L_perfect)
        L_noisy = spatial_consistency_loss(keypoint_coordinates=noisy_kpts)
        print(L_noisy)
        # print(spatial_consistency_loss(keypoint_coordinates=laggy_kpts))
        # print(spatial_consistency_loss(keypoint_coordinates=changing_kpts))
        assert L_perfect < L_noisy

    def test_patch_contrastive_metric(self):
        self.read_perfect()
        self.read_noisy()
        # self.read_laggy()
        self.read_changing()
        perfect_kpts = torch.cat([self.perfect_kpts_1, self.perfect_kpts_2, self.perfect_kpts_3], dim=2)
        perfect_kpts = self.convert_kpts(kpts=perfect_kpts,
                                         H=self.img_sequence_perfect_kpts.shape[3],
                                         W=self.img_sequence_perfect_kpts.shape[4])
        noisy_kpts = torch.cat([self.noisy_kpts_1, self.noisy_kpts_2, self.noisy_kpts_3], dim=2)
        noisy_kpts = self.convert_kpts(kpts=noisy_kpts,
                                       H=self.img_sequence_noisy_kpts.shape[3],
                                       W=self.img_sequence_noisy_kpts.shape[4])
        """
        laggy_kpts = torch.cat([self.laggy_kpts_1, self.laggy_kpts_2, self.laggy_kpts_3], dim=2)
        laggy_kpts = self.convert_kpts(kpts=laggy_kpts,
                                       H=self.img_sequence_laggy_kpts.shape[3],
                                       W=self.img_sequence_laggy_kpts.shape[4])
        """
        changing_kpts = torch.cat([self.changing_kpts_1, self.changing_kpts_2, self.changing_kpts_3], dim=2)
        changing_kpts = self.convert_kpts(kpts=changing_kpts,
                                          H=self.img_sequence_changing_kpts.shape[3],
                                          W=self.img_sequence_changing_kpts.shape[4])

        L_perfect = patchwise_contrastive_metric(image_sequence=self.img_sequence_perfect_kpts,
                                                 kpt_sequence=perfect_kpts,
                                                 time_window=100,
                                                 patch_size=(128, 128),
                                                 alpha=0.1)
        L_noisy = patchwise_contrastive_metric(image_sequence=self.img_sequence_noisy_kpts,
                                               kpt_sequence=noisy_kpts,
                                               time_window=100,
                                               patch_size=(128, 128),
                                               alpha=0.1)
        L_changing = patchwise_contrastive_metric(image_sequence=self.img_sequence_changing_kpts,
                                                  kpt_sequence=changing_kpts,
                                                  time_window=100,
                                                  patch_size=(128, 128),
                                                  alpha=0.1)
        """
        print(patchwise_contrastive_metric(image_sequence=self.img_sequence_laggy_kpts,
                                           kpt_sequence=laggy_kpts,
                                           time_window=9,
                                           patch_size=(25, 25),
                                           alpha=1.0))
        
        """

        print(L_perfect)
        print(L_noisy)
        print(L_changing)

        assert L_perfect < L_noisy < L_changing

    def test_pc_loss(self):
        self.read_perfect()
        self.read_noisy()
        self.read_changing()

        perfect_kpts = torch.cat([self.perfect_kpts_1, self.perfect_kpts_2, self.perfect_kpts_3], dim=2)
        perfect_kpts = self.convert_kpts(kpts=perfect_kpts,
                                         H=self.img_sequence_perfect_kpts.shape[3],
                                         W=self.img_sequence_perfect_kpts.shape[4])

        noisy_kpts = torch.cat([self.noisy_kpts_1, self.noisy_kpts_2, self.noisy_kpts_3], dim=2)
        noisy_kpts = self.convert_kpts(kpts=noisy_kpts,
                                       H=self.img_sequence_noisy_kpts.shape[3],
                                       W=self.img_sequence_noisy_kpts.shape[4])

        changing_kpts = torch.cat([self.changing_kpts_1, self.changing_kpts_2, self.changing_kpts_3], dim=2)
        changing_kpts = self.convert_kpts(kpts=changing_kpts,
                                          H=self.img_sequence_changing_kpts.shape[3],
                                          W=self.img_sequence_changing_kpts.shape[4])

        L_perfect = pixelwise_contrastive_loss(keypoint_coordinates=perfect_kpts,
                                               image_sequence=self.img_sequence_perfect_kpts,
                                               feature_map_seq=...,
                                               alpha=0.1,
                                               time_window=5,
                                               verbose=False)


if __name__ == "__main__":
    unittest.main()

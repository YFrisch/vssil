import os
from os.path import join
import time

import torch
from torch.nn.functional import interpolate
from torchvision.io import read_video

from .mime_base import MimeBase


class MimeHDKinectRGB(MimeBase):

    def __init__(self,
                 base_path: str = join(os.getcwd(), '/datasets'),
                 tasks: str = 'stir',
                 start_ind: int = 0,
                 stop_ind: int = -1,
                 timesteps_per_sample: int = -1,
                 overlap: int = 20,
                 img_shape: (int, int) = (-1, -1)
                 ):

        """ Creates class instance.

        :param img_shape: Desired shape of the images.
            Sampled images are down-/up-sampled to this shape.
            Set to (-1, -1) to use original sizes.
        """

        self.img_shape = img_shape

        print(f"##### Loading MIME dataset of HD Kinect RGB images for task '{tasks}'.")
        time.sleep(0.1)

        super(MimeHDKinectRGB, self).__init__(sample_file_name="hd_kinect_rgb.mp4",
                                              base_path=base_path,
                                              tasks=tasks,
                                              name="hd_kinect_rgb",
                                              start_ind=start_ind,
                                              stop_ind=stop_ind,
                                              timesteps_per_sample=timesteps_per_sample,
                                              overlap=overlap)

        print("##### Done.\n")

    def read_sample(self, path: str) -> (torch.Tensor, int):

        hd_kinect_img_series = None

        try:

            # NOTE: The following are in (T, H, W, C) format
            hd_kinect_img_series = read_video(path, pts_unit='sec')[0]
            hd_kinect_img_series = hd_kinect_img_series.permute(0, 3, 1, 2)
            hd_kinect_img_series = hd_kinect_img_series.float() / 255.0

            if self.img_shape != (-1, -1):
                H = hd_kinect_img_series.shape[2]
                W = hd_kinect_img_series.shape[3]
                H_scale = self.img_shape[0]/H
                W_scale = self.img_shape[1]/W
                hd_kinect_img_series = interpolate(hd_kinect_img_series,
                                                   scale_factor=(H_scale, W_scale),
                                                   recompute_scale_factor=False)

        except RuntimeError as e:
            print(f"\n\n\nCould not read hd kinect data at {path}:")
            print(e)
            print("\n\n\n")

        return hd_kinect_img_series, hd_kinect_img_series.shape[0]

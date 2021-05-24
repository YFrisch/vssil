import time

import torch
from torch.nn.functional import interpolate
from torchvision.io import read_video

from .mime_base import MimeBase


class MimeHDKinectRGB(MimeBase):

    def __init__(self,
                 base_path: str = '/home/yannik/vssil/data/datasets',
                 task: str = 'stir',
                 start_ind: int = 0,
                 stop_ind: int = -1,
                 timesteps_per_sample: int = -1,
                 overlap: int = 20,
                 img_scale_factor: float = 1.0
                 ):

        """ Creates class instance.

        :param img_scale_factor: Factor to down-/upsample frames.
        """

        self.scale_factor = img_scale_factor

        print(f"##### Loading MIME dataset of HD Kinect RGB images for task '{task}'.")
        time.sleep(0.1)

        super(MimeHDKinectRGB, self).__init__(sample_file_name="hd_kinect_rgb.mp4",
                                              base_path=base_path,
                                              task=task,
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
            if self.scale_factor != 1.0:
                hd_kinect_img_series = interpolate(hd_kinect_img_series,
                                                   scale_factor=self.scale_factor,
                                                   recompute_scale_factor=False)

        except RuntimeError as e:
            print(f"\n\n\nCould not read hd kinect data at {path}:")
            print(e)
            print("\n\n\n")

        return hd_kinect_img_series, hd_kinect_img_series.shape[0]

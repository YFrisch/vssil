import ast
from os import listdir
from os.path import isdir, join

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torchvision.io import read_video


class MimeDataSet(Dataset):

    def __init__(self,
                 base_path: str = '/home/yannik/vssil/data/datasets',
                 task: (str, list) = 'stir',
                 joint_data: bool = True,
                 hd_kinect_img_data: bool = False,
                 rd_kinect_img_data: bool = False,
                 rd_sk_left_img_data: bool = False,
                 rd_sk_right_img_data: bool = False):
        """ Creates class instance

        :param base_path: Path to the super folder of all MIME tasks
        :param task: Task or list of tasks to load
        :param joint_data: Whether or not to use joint data
        :param hd_kinect_img_data: Whether or not to use hd kinect image data
        """
        self.base_path = join(base_path, 'mime_' + task)
        self.base_paths = [join(self.base_path, sub_path) for sub_path in listdir(self.base_path)]

        assert isdir(self.base_path), f"Can not read {self.base_path}"

        self.joint_data = joint_data
        self.hd_kinect_img_data = hd_kinect_img_data
        self.rd_kinect_img_data = rd_kinect_img_data
        self.rd_sk_right_img_data = rd_sk_right_img_data
        self.rd_sk_left_img_data = rd_sk_left_img_data

        self.joint_angle_paths = []
        self.left_gripper_paths = []
        self.right_gripper_paths = []
        self.hd_kinect_depth_paths = []
        self.rd_kinect_depth_paths = []
        self.hd_kinect_rgb_paths = []
        self.rd_kinect_rgb_paths = []
        self.rd_sk_left_depth_paths = []
        self.rd_sk_right_depth_paths = []
        self.rd_sk_left_rgb_paths = []
        self.rd_sk_right_rgb_paths = []

        self.read_file_paths()

        self.joint_header = ['left_w0', 'left_w1', 'left_w2', 'right_s0', 'right_s1', 'right_w0',
                             'right_w1', 'head_pan', 'right_w2', 'head_nod', 'torso_t0', 'left_e0',
                             'left_e1', 'left_s0', 'left_s1', 'right_e0', 'right_e1']

    def read_file_paths(self):
        """ Reads file paths."""
        paths = []
        for sub_dir in listdir(self.base_path):
            if self.joint_data:
                self.joint_angle_paths.append(join(self.base_path, sub_dir, "joint_angles.txt"))
                self.left_gripper_paths.append(join(self.base_path, sub_dir, "left_gripper.txt"))
                self.right_gripper_paths.append(join(self.base_path, sub_dir, "right_gripper.txt"))
            if self.hd_kinect_img_data:
                self.hd_kinect_rgb_paths.append(join(self.base_path, sub_dir, "hd_kinect_rgb.mp4"))
                self.hd_kinect_depth_paths.append(join(self.base_path, sub_dir, "hd_kinect_depth.mp4"))
            if self.rd_kinect_img_data:
                self.rd_kinect_rgb_paths.append(join(self.base_path, sub_dir, "rd_kinect_rgb.mp4"))
                self.rd_kinect_depth_paths.append(join(self.base_path, sub_dir, "rd_kinect_depth.mp4"))
            if self.rd_sk_left_img_data:
                self.rd_sk_left_rgb_paths.append(join(self.base_path, sub_dir, "RD_sk_left_rgb.mp4"))
                self.rd_sk_left_depth_paths.append(join(self.base_path, sub_dir, "RD_sk_left_depth.mp4"))
            if self.rd_sk_right_img_data:
                self.rd_sk_right_rgb_paths.append(join(self.base_path, sub_dir, "RD_sk_right_rgb.mp4"))
                self.rd_sk_right_depth_paths.append(join(self.base_path, sub_dir, "RD_sk_right_depth.mp4"))

    def read_joint_data(self, index: int):
        """ Reads the joint-angle data and end effector state for the given index. """
        joint_tensor = None
        try:
            joint_tensor = torch.empty(size=(1, len(self.joint_header) + 2))
            with open(self.joint_angle_paths[index]) as joint_angles_file:
                with open(self.left_gripper_paths[index]) as left_gripper_file:
                    with open(self.right_gripper_paths[index]) as right_gripper_file:
                        left_gripper_lines = left_gripper_file.readlines()
                        right_gripper_lines = right_gripper_file.readlines()

                        for lined_id, line in enumerate(joint_angles_file.readlines()):
                            joint_dict = ast.literal_eval(line)
                            _joint_tensor = torch.tensor(list(joint_dict.values()).append(
                                [left_gripper_lines[lined_id], right_gripper_lines[lined_id]]
                            )).unsqueeze(0)
                            joint_tensor = torch.cat([joint_tensor, _joint_tensor])

        except RuntimeError:
            print(f"Could not read joint-angles at index {index}, removing it.")
            self.base_paths.pop(index)

        return joint_tensor

    def read_hd_kinect_data(self, index: int):
        hd_kinect_img_series, hd_kinect_depth_series = None, None
        try:
            # NOTE: The following are in (T, H, W, C) format
            hd_kinect_img_series = read_video(self.hd_kinect_rgb_paths[index])[0]
            hd_kinect_img_series = hd_kinect_img_series.permute(0, 3, 1, 2)
            hd_kinect_depth_series = read_video(self.hd_kinect_depth_paths[index])[0]
            hd_kinect_depth_series = hd_kinect_depth_series.permute(0, 3, 1, 2)
        except RuntimeError:
            print(f"Could not read hd kinect data at index {index}, removing it.")
            self.base_paths.pop(index)

        return hd_kinect_img_series, hd_kinect_depth_series

    def __getitem__(self, index) -> T_co:

        joint_tensor = torch.empty((1, 1))
        hd_kinect_img_series = torch.empty((1, 1))
        hd_kinect_depth_series = torch.empty((1, 1))
        rd_kinect_img_series = torch.empty((1, 1))
        rd_kinect_depth_series = torch.empty((1, 1))
        rd_sk_left_img_series = torch.empty((1, 1))
        rd_sk_left_depth_series = torch.empty((1, 1))
        rd_sk_right_img_series = torch.empty((1, 1))
        rd_sk_right_depth_series = torch.empty((1, 1))

        if self.joint_data:
            joint_tensor = self.read_joint_data(index)
            assert joint_tensor is not None

        if self.hd_kinect_img_data:
            hd_kinect_img_series, hd_kinect_depth_series = self.read_hd_kinect_data(index)
            assert (hd_kinect_img_series is not None and hd_kinect_depth_series is not None)

        return {"joint_tensor": joint_tensor,
                "hd_kinect_img_series": hd_kinect_img_series, "hd_kinect_depth_series": hd_kinect_depth_series,
                "rd_kinect_img_series": rd_kinect_img_series, "rd_kinect_depth_series": rd_kinect_depth_series,
                "rd_sk_left_img_series": rd_sk_left_img_series, "rd_sk_left_depth_series": rd_sk_left_depth_series,
                "rd_sk_right_img_series": rd_sk_right_img_series, "rd_sk_right_depth_series": rd_sk_right_depth_series}

    def __len__(self):
        return len(self.base_paths)

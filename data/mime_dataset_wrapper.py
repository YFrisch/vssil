from os import listdir
from os.path import isdir, join

import cv2
import ast
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class MimeDataSet(Dataset):

    def __init__(self,
                 base_path: str = '/home/yannik/vssil/data/datasets',
                 task: str = 'stir'):
        self.base_path = join(base_path, 'mime_' + task)
        assert isdir(self.base_path), f"Can not read {self.base_path}"
        self.file_paths = self.read_file_paths()
        self.joint_header = ['left_w0', 'left_w1', 'left_w2', 'right_s0', 'right_s1', 'right_w0',
                                               'right_w1', 'head_pan', 'right_w2', 'head_nod', 'torso_t0', 'left_e0',
                                               'left_e1', 'left_s0', 'left_s1', 'right_e0', 'right_e1']

    def read_file_paths(self) -> list:
        """ TODO: Handle videos

        :return:
        """
        paths = []
        for sub_dir in listdir(self.base_path):
            joined_path = join(self.base_path, sub_dir, "joint_angles.txt")
            paths.append(joined_path)
        return paths

    def __getitem__(self, index) -> T_co:
        try:
            joint_tensor = torch.empty(size=(1, len(self.joint_header)))
            with open(self.file_paths[index]) as joint_angles_file:
                for lined_id, line in enumerate(joint_angles_file.readlines()):
                    joint_dict = ast.literal_eval(line)
                    _joint_tensor = torch.tensor(list(joint_dict.values())).unsqueeze(0)
                    joint_tensor = torch.cat([joint_tensor, _joint_tensor])
        except RuntimeError:
            raise IOError(f"Can not read {self.file_paths[index]}")

        return joint_tensor

    def __len__(self):
        return len(self.file_paths)

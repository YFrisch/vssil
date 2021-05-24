""" This script is used to test some functionalities of the MIME dataset loader / wrapper."""
import gc
import psutil

import torch.cuda
from torch.utils.data import DataLoader, ConcatDataset

from src.data.mime import MimeHDKinectRGB, MimeJointAngles

data_set1 = MimeHDKinectRGB(
    base_path='/home/yannik/vssil/datasets/',
    task='bottle',
    start_ind=0,
    stop_ind=-1,  # Set to -1 to use all available samples
    timesteps_per_sample=-1,  # Set to -1 to return whole trajectory
    overlap=0,
    img_scale_factor=0.25
)

data_set2 = MimeHDKinectRGB(
    base_path='/home/yannik/vssil/datasets/',
    task='stir',
    start_ind=0,
    stop_ind=-1,  # Set to -1 to use all available samples
    timesteps_per_sample=-1,  # Set to -1 to return whole trajectory
    overlap=0,
    img_scale_factor=0.25
)

joint_angles_data_set = MimeJointAngles(
    base_path='/home/yannik/vssil/datasets/',
    task='stir',
    start_ind=0,
    stop_ind=-1,
    timesteps_per_sample=-1,
    overlap=0
)

data_set = ConcatDataset([data_set1, data_set2])

data_loader = DataLoader(
    data_set,
    batch_size=1,
    shuffle=True
)

joint_angles_data_loader = DataLoader(
    joint_angles_data_set,
    batch_size=1,
    shuffle=True
)

dl = joint_angles_data_loader

for i, sample in enumerate(dl):

    # sample = sample.to("cuda:0")
    print(f"Sample: {i + 1}|{len(dl)}\t "
          f"Shape: {sample.shape}\t "
          #f"Alloc. GPU mem: {torch.cuda.memory_allocated()}\t "
          #f"Used RAM: {psutil.virtual_memory().used / 1e+9}|{psutil.virtual_memory().total / 1e+9}\t "
          #f"Used SWAP: {psutil.swap_memory().used / 1e+9}|{psutil.swap_memory().total / 1e+9}"
    )

    del sample
    gc.collect()


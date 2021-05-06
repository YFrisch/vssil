""" This script is used to test some functionalities of the MIME dataset loader / wrapper."""
import gc
import psutil

import torch.cuda
from torch.utils.data import DataLoader, ConcatDataset

from src.data.mime import MimeHDKinectRGB

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

data_set = ConcatDataset([data_set1, data_set2])

data_loader = DataLoader(
    data_set,
    batch_size=1,
    shuffle=True
)

print(f"Loaded data-set of {sum([len(data_set) for data_set in data_set.datasets])} "
      f"trajectories and {len(data_loader)} samples.")

for i, sample in enumerate(data_loader):

    # sample = sample.to("cuda:0")
    print(f"Sample: {i + 1}|{len(data_loader)}\t "
          f"Shape: {sample.shape}\t "
          #f"Alloc. GPU mem: {torch.cuda.memory_allocated()}\t "
          #f"Used RAM: {psutil.virtual_memory().used / 1e+9}|{psutil.virtual_memory().total / 1e+9}\t "
          #f"Used SWAP: {psutil.swap_memory().used / 1e+9}|{psutil.swap_memory().total / 1e+9}"
    )

    del sample
    gc.collect()

